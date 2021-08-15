#!/usr/bin/env python

import argparse
import copy
import pathlib
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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


def attack(config, model, train_loader, test_loader, loss_func, logger):
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
            post_tuned_model = post_tune(config, model, adv_images, train_loader)
            post_tuned_output = post_tuned_model(adv_images)
            print()
            print("adv ", adv_output)
            print("post", post_tuned_output)
            print(torch.argmax(post_tuned_output), labels)
            # acc = cal_accuracy(normal_output, labels)
            # acc = cal_accuracy(adv_output, labels)
            acc = cal_accuracy(post_tuned_output, labels)
            if attack_target_class == -1:
                success = cal_accuracy(adv_output, attack_target_list[-1])
            elif attack_target_class is not None:
                success = cal_accuracy(adv_output, torch.Tensor([attack_target_class]).to(device))
            else:
                success = 0
            print("Batch {} attack success: {}\tdefense acc: {}\n".format(i, success, acc))
            input()
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


def merge_images(train_images, val_images, device):
    image = 0.9 * train_images.to(device) + 0.1 * val_images.to(device)
    # image[0][channel] = 0.5 * image[0][channel].to(device) + 0.5 * val_images[0][channel].to(device)
    return image


def post_tune(config, model, images, train_loader):
    alpha = 2 / 255
    epsilon = 8 / 255
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    model = copy.deepcopy(model)
    fix_model = copy.deepcopy(model)

    # images = images.detach() + (torch.rand_like(images.detach()) * 2 - 1) * epsilon
    # images = images.detach() + torch.sign(torch.rand_like(images.detach()) * 2 - 1) * epsilon

    original_output = fix_model(images)
    print('original', torch.argmax(original_output), original_output)

    with torch.enable_grad():
        # # re-position start
        # attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=2, random_start=False)
        # original_label = torch.argmax(original_output).reshape(1)
        # images = attack_model(images, original_label)
        # start_output = fix_model(images)
        # print('start', torch.argmax(start_output), start_output)

        # optimizer = create_optimizer(config, model)
        optimizer = torch.optim.SGD(lr=0.001,
                                    params=model.parameters(),
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
        targets = torch.ones([len(images)], dtype=torch.long).to(device) * int(torch.argmax(original_output))
        attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        # targets_list = torch.topk(original_output, k=3).indices.squeeze().detach()
        # target_list = [i for i in range(10)]
        # random.shuffle(target_list)
        # targets_list = torch.Tensor(target_list).long().to(device)

        # find target candidates
        # targets_list = torch.Tensor([-1, -1])
        # targets_list[0] = int(torch.argmax(original_output))
        # adv_inputs = attack_model(images, targets)
        # adv_inputs.requires_grad = True
        # adv_outputs = model(adv_inputs.detach())
        # targets_list[1] = int(torch.argmax(adv_outputs))
        # targets_list = targets_list.long().to(device)
        # print('targets_list', targets_list)

        for _ in range(5):
            loss_list = torch.Tensor([0 for _ in range(10)])
            for i in range(4):
                # train_images, train_label = next(iter(train_loader))
                # images = merge_images(train_images, val_images, device)
                outputs_list = []
                # targets = targets_list[i % len(targets_list)].reshape([1])
                # targets = targets_list[1].reshape([1])  # guess target
                # targets = torch.randint(0, 9, [len(images)]).to(device)
                # targets = train_label.to(device)
                optimizer.zero_grad()
                # noise = ((torch.rand_like(images.detach()) * 2 - 1) * epsilon).to(device)  # uniform rand from [-eps, eps]
                # noise_inputs = images.detach() + noise
                # noise_inputs.requires_grad = True
                # noise_outputs = model(noise_inputs)
                # noise_loss = loss_func(noise_outputs, targets)
                #
                # # loss = loss_func(noise_outputs, targets)  # loss to be maximized
                # input_grad = torch.autograd.grad(noise_loss, noise_inputs)[0]
                # # print(torch.mean(torch.abs(input_grad)))
                # delta = noise - alpha * torch.sign(input_grad)
                # delta.clamp_(-epsilon, epsilon)
                #
                # adv_inputs = images + delta
                # attack_model.set_mode_targeted_by_function(lambda image, label: targets)
                adv_inputs = attack_model(images, targets)
                adv_inputs.requires_grad = True
                outputs = model(adv_inputs.detach())
                adv_loss = loss_func(outputs, targets)

                # update target
                outputs_list.append(outputs)
                normal_output = model(images.detach())
                normal_loss = loss_func(normal_output, targets)
                # loss_list[i] = torch.relu(adv_loss - normal_loss)  # untargeted
                # loss_list[i] = noise_loss - adv_loss  # targeted
                loss_list[i] = adv_loss  # untargeted
                # loss_list[i] = -1 * adv_loss  # targeted
                print(int(targets.item()),
                      '{:.4f}'.format(float(torch.relu(adv_loss - normal_loss))),
                      '{:.4f}'.format(float(adv_loss)),
                      outputs)
                targets = torch.ones([len(images)], dtype=torch.long).to(device) * int(torch.argmax(outputs))
                # print(targets, torch.softmax(outputs, dim=1), torch.softmax(original_output, dim=1))

            # loss = loss_func(outputs, targets)
            # kl_loss = nn.KLDivLoss(size_average=False, log_target=True)(
            #     torch.log_softmax(outputs_list[0], dim=1),
            #     torch.log_softmax(outputs_list[1], dim=1)
            # )
            # amplitude_regularization = torch.sum(torch.abs(outputs_list[0])) + torch.sum(torch.abs(outputs_list[0]))
            # loss = kl_loss + 0 * amplitude_regularization
            # print(loss, kl_loss, 0 * amplitude_regularization)
            # adjusted_loss_list = loss_list - torch.min(loss_list)
            total_loss = torch.mean(torch.Tensor(loss_list))
            # total_loss.requires_grad = True
            loss = total_loss
            print(loss)
            loss.backward()
            optimizer.step()

    return model


def main():
    config = load_config(["train.batch_size", 1, "validation.batch_size", 1, "test.batch_size", 1])

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

    train_loader, test_loader = create_dataloader(config, is_train=True)
    _, test_loss = create_loss(config)

    attack(config, model, train_loader, test_loader, test_loss, logger)


if __name__ == '__main__':
    main()
