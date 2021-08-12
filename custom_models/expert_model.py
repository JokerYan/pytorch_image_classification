import pathlib

import torch
import torch.nn as nn
from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import create_model

config_path = 'configs/cifar/resnet.yaml'
naive_model_path = 'experiments/cifar10/resnet/exp12/checkpoint_00160.pth'
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


class ExpertModel(nn.Module):
    def __init__(self, config):
        super(ExpertModel, self).__init__()
        self.config_list = config
        model_list = []
        for i in range(len(model_path_dict)):
            model_path = model_path_dict[i]
            output_dir = pathlib.Path(model_path).parent
            model = create_model(config)

            # load model
            checkpointer = Checkpointer(model,
                                        save_dir=output_dir)
            checkpointer.load(model_path)

            model_list.append(model)
        self.model_list = nn.ModuleList(model_list)

        # create and load naive model (without adv training)
        self.naive_model = create_model(config)
        model_path = naive_model_path
        output_dir = pathlib.Path(model_path).parent
        checkpointer = Checkpointer(self.naive_model,
                                    save_dir=output_dir)
        checkpointer.load(model_path)

    def forward(self, x):
        naive_output = self.naive_model(x)
        target_class = torch.argmax(naive_output)

        print(target_class)

        expert_output = self.model_list[target_class[0]](x)
        return expert_output

