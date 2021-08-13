#!/usr/bin/env python

import argparse
import copy
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
    update_config, create_optimizer,
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


attack_target_list = []
def random_target_function(images, labels):
    attack_target = torch.remainder(torch.randint(1, 9, labels.shape).cuda() + labels, 10)
    # attack_target = torch.ones_like(labels).cuda() * 9
    global attack_target_list
    attack_target_list.append(attack_target)
    print("===========", len(attack_target_list))
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

        # targeted
        if attack_target_class is not None and attack_target_class != -1:
            attack_target_tensor = torch.ones_like(labels) * attack_target_class
            attack_target_tensor[labels == attack_target_tensor] = (attack_target_class + 1) % config.dataset.n_classes
            attack_model.set_mode_targeted_by_function(lambda images, labels: attack_target_tensor)  # targeted attack
        elif attack_target_class == -1:
            attack_model.set_mode_targeted_by_function(random_target_function)

        adv_images = attack_model(data, labels)

        with torch.no_grad():
            adv_output = model(adv_images)
            normal_output = model(data)
            print("normal_output: {} output: {} labels: {}".format(
                int(torch.argmax(normal_output)),
                int(torch.argmax(adv_output)), int(labels)))

            # post_train(config, model, adv_images, labels)
            post_tuned_model = post_tune(config, model, adv_images)
            post_tuned_output = post_tuned_model(adv_images)
            # acc = cal_accuracy(normal_output, labels)
            # acc = cal_accuracy(adv_output, labels)
            acc = cal_accuracy(post_tuned_output, labels)
            if attack_target_class == -1:
                success = cal_accuracy(adv_output, attack_target_list[-1])
            elif attack_target_class is not None:
                success = cal_accuracy(adv_output, torch.Tensor([attack_target_class]).to(device))
            else:
                success = 0
            print("Batch {} attack success: {}\tdefense acc: {}".format(i, success, acc))
            success_meter.update(success, 1)
            accuracy_meter.update(acc, 1)
        adv_image_list.append(adv_images)

    logger.info(f'Success: {success_meter.avg:.4f} Accuracy {accuracy_meter.avg:.4f} Delta {delta_meter.avg:.4f}')

    return adv_image_list, accuracy_meter.avg, delta_meter.avg


def post_train(config, model, images, targets):
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())
    with torch.enable_grad():
        for i in range(10):
            optimizer.zero_grad()
            outputs = model(images)
            print(targets, torch.argmax(outputs).item(), outputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            input()


def post_tune(config, model, images):
    alpha = 2 / 255
    epsilon = 8 / 255
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    model = copy.deepcopy(model)

    fix_model = copy.deepcopy(model)
    original_output = fix_model(images)
    with torch.enable_grad():
        # optimizer = create_optimizer(config, model)
        optimizer = torch.optim.SGD(lr=0.0001,
                                    params=model.parameters(),
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
        # targets = torch.ones([len(images)], dtype=torch.long).to(device) * int(torch.argmax(original_output))
        attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        for _ in range(10):
            targets = torch.randint(0, 9, [len(images)]).to(device)
            optimizer.zero_grad()
            # noise = (torch.rand_like(images.detach()) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            # noise_inputs = images.detach() + noise
            # noise_inputs.requires_grad = True
            # noise_outputs = model(noise_inputs)
            #
            # loss = loss_func(noise_outputs, targets)  # loss to be maximized
            # input_grad = torch.autograd.grad(loss, noise_inputs)[0]
            # # print(torch.mean(torch.abs(input_grad)))
            # delta = noise + alpha * torch.sign(input_grad)
            # delta.clamp_(-epsilon, epsilon)
            #
            # adv_inputs = images + delta
            adv_inputs = attack_model(images, targets)
            adv_inputs.requires_grad = True
            outputs = model(adv_inputs)
            # print(targets[0], torch.argmax(outputs).item())
            print(targets, torch.softmax(outputs, dim=1), torch.softmax(original_output, dim=1))

            input_grad_norm = torch.autograd.grad(torch.sum(outputs), adv_inputs)[0]
            loss = input_grad_norm
            # loss = loss_func(outputs, targets)
            # loss = nn.KLDivLoss(size_average=False, log_target=True)(
            #     torch.log_softmax(outputs, dim=1),
            #     torch.log_softmax(original_output, dim=1)
            # )
            print(loss)
            loss.backward()
            optimizer.step()
            input()

    return model


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
