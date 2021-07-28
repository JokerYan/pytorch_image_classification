import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, config, base_model):
        super(Network, self).__init__()
        self.temperature = 0.1
        self.config = config
        self.base_model = base_model

    def forward(self, x):
        x_sigmoid = torch.sigmoid(x / self.temperature)
        return self.base_model(x_sigmoid)
