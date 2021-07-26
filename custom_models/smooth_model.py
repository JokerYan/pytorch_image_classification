import torch
import torch.nn as nn


class SmoothModel(nn.Module):
    def __init__(self, base_model, mean=0, std=0.1, sample_size=100):
        super(SmoothModel, self).__init__()
        self.base_model = base_model
        self.mean = mean
        self.std = std
        self.sample_size = sample_size

    def forward(self, x):
        base_output = self.base_model(x)
        input_dummy = torch.ones(x.shape)
        output_list = []
        for i in range(self.sample_size):
            gaussian_noise = torch.normal(self.mean * input_dummy, self.std * input_dummy)
            gaussian_input = x + gaussian_noise
            gaussian_output = self.base_model(gaussian_input)
            output_list.append(gaussian_output)
        return torch.mean(torch.stack(output_list), dim=0)
