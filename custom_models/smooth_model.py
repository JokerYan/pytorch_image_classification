import torch
import torch.nn as nn

from utils.debug_tools import save_image_stack, clear_debug_image


class SmoothModel(nn.Module):
    def __init__(self, base_model, mean=0, std=0.1, sample_size=10):
        super(SmoothModel, self).__init__()
        self.base_model = base_model
        self.mean = mean
        self.std = std
        self.sample_size = sample_size

    def forward(self, x):
        input_clone = x.clone().detach()
        input_clone.requires_grad = True
        base_output = self.base_model(input_clone)
        torch.max(base_output).backward()

        grad_data = input_clone.grad.data
        grad_data -= grad_data.min(1, keepdim=True).values
        grad_data /= grad_data.max(1, keepdim=True).values

        input_dummy = torch.ones(x.shape)
        output_list = []
        for i in range(self.sample_size):
            gaussian_noise = torch.normal(self.mean * input_dummy, self.std * input_dummy).cuda()

            gaussian_noise = gaussian_noise * grad_data

            gaussian_input = x + gaussian_noise
            gaussian_output = self.base_model(gaussian_input)
            output_list.append(gaussian_output)
        return torch.mean(torch.stack(output_list), dim=0)
