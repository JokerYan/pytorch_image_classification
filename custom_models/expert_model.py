import pathlib

import torch
import torch.nn as nn
from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import create_model

config_path = 'configs/cifar/resnet.yaml'
naive_model_path = 'experiments/cifar10/resnet/exp12/checkpoint_00160.pth'
untargetd_model_path = 'experiments/cifar10/resnet/exp09/checkpoint_00160.pth'
model_path_dict = {
    0: 'experiments/cifar10/resnet/exp11/checkpoint_00160.pth',
    1: 'experiments/cifar10/resnet/exp13/checkpoint_00160.pth',
    2: 'experiments/cifar10/resnet/exp14/checkpoint_00160.pth',
    3: 'experiments/cifar10/resnet/exp15/checkpoint_00160.pth',
    4: 'experiments/cifar10/resnet/exp16/checkpoint_00160.pth',
    5: 'experiments/cifar10/resnet/exp17/checkpoint_00160.pth',
    6: 'experiments/cifar10/resnet/exp18/checkpoint_00160.pth',
    7: 'experiments/cifar10/resnet/exp19/checkpoint_00160.pth',
    # 8: 'experiments/cifar10/resnet/exp20/checkpoint_00160.pth',
    # 9: 'experiments/cifar10/resnet/exp21/checkpoint_00160.pth',
}


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config_list = config
        model_list = []
        for i in range(len(model_path_dict)):
            model_path = model_path_dict[i]
            model = self.create_model_from_path(config, model_path)
            model_list.append(model)
        self.model_list = nn.ModuleList(model_list)

        self.naive_model = self.create_model_from_path(config, naive_model_path)
        self.untargetd_model = self.create_model_from_path(config, untargetd_model_path)

    def create_model_from_path(self, config, model_path):
        output_dir = pathlib.Path(model_path).parent
        model = create_model(config)

        # load model
        checkpointer = Checkpointer(model,
                                    save_dir=output_dir)
        checkpointer.load(model_path)
        return model

    def forward(self, x):
        naive_output = self.naive_model(x)
        target_class = torch.argmax(naive_output).item()

        if target_class in model_path_dict:
            expert_output = self.model_list[target_class](x)
        else:
            expert_output = self.untargetd_model(x)
        print(target_class, torch.argmax(expert_output).item())
        return expert_output
        # return naive_output

