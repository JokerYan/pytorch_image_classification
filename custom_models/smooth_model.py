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
        base_output = self.base_model(x)
        x.retain_grad()
        torch.max(base_output).backward()
        print(x.requires_grad)
        print(x.grad.data)
        x.grad.data.zero_()

        input_dummy = torch.ones(x.shape)
        output_list = []
        for i in range(self.sample_size):
            gaussian_noise = torch.normal(self.mean * input_dummy, self.std * input_dummy).cuda()
            gaussian_input = x + gaussian_noise
            gaussian_output = self.base_model(gaussian_input)
            output_list.append(gaussian_output)
        return torch.mean(torch.stack(output_list), dim=0)
