#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import torchvision.transforms as torch_transforms
import tqdm

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.models import create_input_sigmoid_model, create_expert_model
from pytorch_image_classification.transforms import _get_dataset_stats
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)
from custom_torchattacks.custom_cw import CustomCW
from custom_torchattacks.custom_pgd import CustomPGD
from utils.debug_tools import clear_debug_image, save_image_stack


attack_target_class = None

def load_config(options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--target', type=int, default=None)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if options:
        config.merge_from_list(options)
    if args.target is not None:
        global attack_target_class
        attack_target_class = args.target
    update_config(config)
    config.freeze()
    return config


def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total


def random_target_function(images, labels):
    attack_target = torch.remainder(torch.randint(1, 9, labels.shape).cuda() + labels, 10)
    print(attack_target)
    # attack_target = torch.ones_like(labels).cuda() * 9
    return attack_target


def attack(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)
    print(torchattacks.__file__)

    model.eval()
    # attack_model = CWInfAttack(model, config, c, lr, momentum, steps).cuda()
    # attack_model = torchattacks.CW(model, c=1, steps=1000, lr=0.01)
    # attack_model = CustomCW(config, model, c=1, steps=200, lr=0.01)
    # attack_model.set_mode_targeted_by_function(random_target_function)
    attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
    # attack_model = CustomPGD(config, model, eps=8/255, alpha=2/255, steps=20)

    success_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    delta_meter = AverageMeter()

    adv_image_list = []

    mean, std = _get_dataset_stats(config)
    Normalize = torch_transforms.Normalize(
        mean, std
    )

    for i, (data, labels) in enumerate(test_loader):
        if i == 100:
            break
        data = data.to(device)
        labels = labels.to(device)

        # data = Normalize(data)

        adv_images = attack_model(data, labels)
        # targeted
        if attack_target_class is not None:
            attack_target_tensor = torch.ones_like(labels) * attack_target_class
            attack_target_tensor[labels == attack_target_tensor] = (attack_target_class + 1) % config.dataset.n_classes
            attack_model.set_mode_targeted_by_function(lambda images, labels: attack_target_tensor)  # targeted attack
        else:
            attack_model.set_mode_targeted_by_function(random_target_function)

        with torch.no_grad():
            adv_output = model(adv_images)
            normal_output = model(data)
            # success = cal_accuracy(adv_output, labels)
            print("normal_output: {} output: {} labels: {}".format(
                int(torch.argmax(normal_output)),
                int(torch.argmax(adv_output)), int(labels)))
            acc = cal_accuracy(adv_output, labels)
            # acc = cal_accuracy(normal_output, labels)
            print("Batch {} attack success: {}\tdefense acc: {}".format(i, "N.A.", acc))
            # success_meter.update(success, 1)
            accuracy_meter.update(acc, 1)
        adv_image_list.append(adv_images)

    logger.info(f'Success: {success_meter.avg:.4f} Accuracy {accuracy_meter.avg:.4f} Delta {delta_meter.avg:.4f}')

    return adv_image_list, accuracy_meter.avg, delta_meter.avg


def main():
    config = load_config(["test.batch_size", 1])

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    if config.custom_model.name == 'input_sigmoid_model':
        model = create_input_sigmoid_model(config)
    elif config.custom_model.name == 'expert_model':
        model = create_expert_model(config)
    else:
        model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)

    if config.custom_model.name != 'expert_model':
        checkpointer = Checkpointer(model,
                                    save_dir=output_dir)
        checkpointer.load(config.test.checkpoint)

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    attack(config, model, test_loader, test_loss, logger)


if __name__ == '__main__':
    main()
