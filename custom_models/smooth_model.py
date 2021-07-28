import torch
import torch.nn as nn

from utils.debug_tools import save_image_stack, clear_debug_image


class SmoothModel(nn.Module):
    def __init__(self, base_model, mean=0, std=0.5, sample_size=20):
        super(SmoothModel, self).__init__()
        self.base_model = base_model
        self.mean = mean
        self.std = std
        self.sample_size = sample_size

    def forward(self, x):
        input_clone = x.clone().detach()
        input_clone.requires_grad = True
        base_output = self.base_model(input_clone)

        # torch.max(base_output).backward()
        # grad_data = input_clone.grad.data
        # grad_data = torch.abs(grad_data)
        # grad_data -= grad_data.min(1, keepdim=True).values
        # grad_data /= grad_data.max(1, keepdim=True).values

        input_dummy = torch.ones(x.shape)
        output_list = []
        output_c_list = []
        for i in range(self.sample_size):
            gaussian_noise = torch.normal(self.mean * input_dummy, self.std * input_dummy).cuda()
            # linear_noise = torch.randn_like(x).cuda() * 0.1 + 0.9

            # gaussian_noise = gaussian_noise * grad_data
            gaussian_noise = gaussian_noise * self.get_focus_filter(x.shape)
            save_image_stack(torch.mean(torch.abs(gaussian_noise), dim=1, keepdim=True), "gaussian_noise_{}".format(i))

            gaussian_input = x + gaussian_noise
            save_image_stack(gaussian_input, "gaussian_input_{}".format(i), normalized=True)

            # gaussian_input = x * linear_noise
            gaussian_output = self.base_model(gaussian_input)
            # min max norm to 0 ~ 1
            gaussian_output -= gaussian_output.min(1, keepdim=True).values
            gaussian_output /= gaussian_output.max(1, keepdim=True).values

            output_list.append(gaussian_output)
            output_c_list.append(int(torch.max(gaussian_output, dim=1).indices))
        print(output_c_list)
        return torch.mean(torch.stack(output_list), dim=0)

    def get_focus_filter(self, shape):
        max_distance = 16
        # shape: Batch x Channel x H x W
        focus_filter = torch.ones(shape)
        h_center = torch.randint(0, shape[2], (1, ))
        w_center = torch.randint(0, shape[3], (1, ))
        # print(shape, h_center, w_center)

        for b in range(focus_filter.shape[0]):
            for c in range(focus_filter.shape[1]):
                for h in range(focus_filter.shape[2]):
                    for w in range(focus_filter.shape[3]):
                        distance_to_center = torch.sqrt(torch.square(h - h_center) + torch.square(w - w_center))
                        focus_filter[b][c][h][w] = 1 - min(1, distance_to_center / max_distance)
        save_image_stack(focus_filter, "focus_filter")
        return focus_filter.cuda()

