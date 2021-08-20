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
import torchvision.transforms
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
from pytorch_image_classification.datasets.dataloader import create_dataloader_by_class
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


def attack(config, model, train_loader, test_loader, train_loaders_by_class, loss_func, logger):
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
    post_accuracy_meter = AverageMeter()
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

            # test_random(config, model, data)
            # post_test(config, model, adv_images, data, labels)
            # post_tuned_model = post_tune(config, model, adv_images, train_loader)
            post_trained_model = post_train(config, model, adv_images, train_loaders_by_class)
            # post_tuned_output = post_tuned_model(adv_images)
            post_trained_output = post_trained_model(adv_images)
            print()
            print("adv ", adv_output)
            print("post", post_trained_output)
            print(torch.argmax(post_trained_output), labels)
            # # acc = cal_accuracy(normal_output, labels)
            acc = cal_accuracy(adv_output, labels)
            post_acc = cal_accuracy(post_trained_output, labels)
            # acc = cal_accuracy(post_output, labels)
            if attack_target_class == -1:
                success = cal_accuracy(adv_output, attack_target_list[-1])
            elif attack_target_class is not None:
                success = cal_accuracy(adv_output, torch.Tensor([attack_target_class]).to(device))
            else:
                success = 0
            success_meter.update(success, 1)
            accuracy_meter.update(acc, 1)
            post_accuracy_meter.update(post_acc, 1)
            print("Batch {} success: {}\tacc: {}({:.4f})\tpost acc: {}({:.4f})\n".format(
                i, success, acc, accuracy_meter.avg, post_acc, post_accuracy_meter.avg))
            # input()
        adv_image_list.append(adv_images)

    logger.info(f'Success: {success_meter.avg:.4f} Accuracy {accuracy_meter.avg:.4f} Delta {delta_meter.avg:.4f}')

    return adv_image_list, accuracy_meter.avg, delta_meter.avg


def test_random(config, model, image):
    epsilon = 8 / 255
    print('ori', int(torch.argmax(model(image))))
    transform = torch_transforms.RandomResizedCrop(size=32, scale=(1/10, 1/8))
    # random_image = torch.rand_like(image)
    random_image = transform(image)
    print(random_image[0][0])
    output = model(random_image)
    targets = torch.argmax(output, dim=1)
    print(int(targets))
    attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
    with torch.enable_grad():
        for i in range(10):
            adv_image = attack_model(random_image, targets)
            adv_output = model(adv_image)
            adv_class = torch.argmax(adv_output, dim=1)
            print(int(adv_class))
    input()


def transform_image(image):
    transform = torch_transforms.Compose([
        # torch_transforms.RandomHorizontalFlip(),
        # torch_transforms.RandomVerticalFlip(),
        torch_transforms.RandomResizedCrop(size=32, scale=(1/8, 1/4)),
        # torch_transforms.GaussianBlur(kernel_size=9),
        # torch_transforms.RandomAutocontrast(),
    ])
    return transform(image)


total_counter = 0
neighbour_counter = 0
mode_counter = 0
def post_test(config, model, images, normal_images, labels):
    alpha = 2 / 255
    epsilon = 8 / 255
    attack_model = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    model = copy.deepcopy(model)
    # optimizer = torch.optim.SGD(lr=0.0001, params=model.parameters())
    with torch.enable_grad():
        initial_output = model(images)
        initial_class = torch.argmax(initial_output).long().reshape(1)

        neighbour_images = attack_model(images, initial_class)
        neighbour_output = model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).long().reshape(1)

        normal_output = model(normal_images)

        trans_normal_images = transform_image(normal_images)
        trans_initial_images = transform_image(images)
        trans_neighbour_images = transform_image(neighbour_images)

        trans_normal_output = model(trans_normal_images)
        trans_initial_output = model(trans_initial_images)
        trans_neighbour_output = model(trans_neighbour_images)

        trans_normal_class = int(torch.argmax(trans_normal_output))
        trans_initial_class = int(torch.argmax(trans_initial_output))
        trans_neighbour_class = int(torch.argmax(trans_neighbour_output))

        kl_loss_normal = nn.KLDivLoss(size_average=False, log_target=True)(
                    torch.log_softmax(trans_normal_output, dim=1),
                    torch.log_softmax(normal_output, dim=1)
                )
        kl_loss_initial = nn.KLDivLoss(size_average=False, log_target=True)(
                    torch.log_softmax(trans_initial_output, dim=1),
                    torch.log_softmax(initial_output, dim=1)
                )
        kl_loss_neighbour = nn.KLDivLoss(size_average=False, log_target=True)(
                    torch.log_softmax(trans_neighbour_output, dim=1),
                    torch.log_softmax(neighbour_output, dim=1)
                )

        print('{}:{:.5f}  {}:{:.5f}  {}:{:.5f}'.format(
            trans_normal_class, kl_loss_normal,
            trans_initial_class, kl_loss_initial,
            trans_neighbour_class, kl_loss_neighbour))

        # middle_images = (images.detach() + neighbour_images.detach()) / 2
        # noise = 0 * ((torch.rand_like(middle_images.detach()) * 2 - 1) * epsilon).to(device)  # uniform rand from [-eps, eps]
        # noise_inputs = middle_images.detach() + noise
        # noise_inputs.requires_grad = True
        # noise_outputs = model(noise_inputs)
        # noise_loss_normal = loss_func(noise_outputs, initial_class)
        # noise_loss_neighbour = loss_func(noise_outputs, neighbour_class)
        # noise_loss = (noise_loss_normal + noise_loss_neighbour) / 2
        # input_grad = torch.autograd.grad(noise_loss, noise_inputs)[0]
        # delta = noise + alpha * torch.sign(input_grad)
        # delta.clamp_(-epsilon, epsilon)
        # adv_images = middle_images + delta
        # adv_output = model(adv_images)
        # adv_class = torch.argmax(adv_output)

        # mix_class_list = []
        # mix_class_correct_list = []
        # for i in [0.1 * x for x in range(1, 10)]:
        #     noise = ((torch.rand_like(images.detach()) * 2 - 1) * epsilon).to(device)  # uniform rand from [-eps, eps]
        #     mix_images = i * images + (1 - i) * neighbour_images + noise
        #     mix_output = model(mix_images)
        #     mix_class = torch.argmax(mix_output)
        #     mix_class_list.append(mix_class)
        #     mix_class_correct_list.append(1 if int(mix_class) == int(labels) else 0)
        # print(mix_class_correct_list, int(torch.mode(torch.Tensor(mix_class_correct_list)).values))

        global neighbour_counter
        global total_counter
        total_counter += 1
        # if int(torch.mode(torch.Tensor(mix_class_correct_list)).values):
        #     global mode_counter
        #     mode_counter += 1
        if int(labels) == int(initial_class) or int(labels) == int(neighbour_class):
            neighbour_counter += 1
        print('label: {} init: {} neighbour: {}'
              .format(int(labels), int(initial_class), int(neighbour_class)))
        print(neighbour_counter, '/', total_counter)
        # print(mode_counter, '/', total_counter)
        # input()


def merge_images(train_images, val_images, device):
    image = 0.9 * train_images.to(device) + 0.1 * val_images.to(device)
    # image[0][channel] = 0.5 * image[0][channel].to(device) + 0.5 * val_images[0][channel].to(device)
    return image


def post_train(config, model, images, train_loaders_by_class):
    alpha = 2 / 255
    epsilon = 8 / 255
    loss_func = nn.CrossEntropyLoss()
    device = torch.device(config.device)
    model = copy.deepcopy(model)
    fix_model = copy.deepcopy(model)
    attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=5)
    optimizer = torch.optim.SGD(lr=0.003,
                                params=model.parameters(),
                                momentum=config.train.momentum,
                                nesterov=config.train.nesterov)
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)
        neighbour_images = attack_model(images, original_class)
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            return model

        for _ in range(5):
            # # reinforce train
            # count_cap = 8
            # effective_count = 0
            # effective_original_count = 0
            # effective_neighbour_count = 0
            # # defense_success = 0
            # # loss_list = torch.Tensor([0 for _ in range(count_cap * 2)]).to(device)
            # input_list = torch.zeros([count_cap*2, 3, 32, 32]).to(device)
            # label_list = torch.zeros([count_cap*2]).to(device)
            # target_list = torch.zeros([count_cap*2]).to(device)
            # while effective_count < count_cap * 2:
            #     data, label = next(iter(train_loader))
            #     data = data.to(device)
            #     label = label.to(device)
            #     if int(label) != int(original_class) and int(label) != int(neighbour_class):
            #         continue
            #     if int(label) == int(original_class):
            #         if effective_original_count > count_cap:
            #             continue
            #         else:
            #             effective_original_count += 1
            #     if int(label) == int(neighbour_class):
            #         if effective_neighbour_count > count_cap:
            #             continue
            #         else:
            #             effective_neighbour_count +=1
            #     effective_count += 1
            #     # targeted attack
            #     target = neighbour_class if int(label) == original_class else original_class
            #     assert target != label
            #     # attack_model.set_mode_targeted_by_function(lambda im, la: target)
            #     # adv_input = attack_model(data, label)
            #     input_list[effective_count - 1] = data
            #     label_list[effective_count - 1] = label
            #     target_list[effective_count - 1] = target
            #
            # data = input_list.detach()
            # label = label_list.long().detach()
            # target = target_list.long().detach()

            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            data = torch.vstack([original_data, neighbour_data])
            label = torch.vstack([original_label, neighbour_label])
            target = torch.vstack([neighbour_label, original_label])

            # generate fgsm adv examples
            delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            noise_input = data + delta
            noise_input.requires_grad = True
            noise_output = model(noise_input)
            loss = loss_func(noise_output, target)  # loss to be maximized
            input_grad = torch.autograd.grad(loss, noise_input)[0]
            delta = delta + alpha * torch.sign(input_grad)
            delta.clamp_(-epsilon, epsilon)
            adv_input = data + delta

            adv_output = model(adv_input.detach())
            # adv_class = torch.argmax(adv_output)
            loss_pos = loss_func(adv_output, label)
            loss_neg = loss_func(adv_output, target)
            # loss_list[effective_count - 1] = loss_pos
            # print(int(label), int(torch.argmax(adv_output)), loss_list[effective_count - 1])

            # loss = torch.mean(loss_list)
            loss = loss_pos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            print('loss: {:.4f}  acc: {:.4f}'.format(loss, defense_acc))
    return model


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
    attack_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)

    with torch.enable_grad():
        # find neighbour
        original_class = torch.argmax(original_output).reshape(1)
        neighbour_images = attack_model(images, original_class)
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)
        #
        # noise = ((torch.rand_like(images.detach()) * 2 - 1) * epsilon).to(device)  # uniform rand from [-eps, eps]
        # noise_inputs = images.detach() + noise
        # noise_inputs.requires_grad = True
        # noise_outputs = model(noise_inputs)
        # noise_loss_normal = loss_func(noise_outputs, original_class)
        # noise_loss_neighbour = loss_func(noise_outputs, neighbour_class)
        # noise_loss = (noise_loss_normal + noise_loss_neighbour) / 2
        # input_grad = torch.autograd.grad(noise_loss, noise_inputs)[0]

        # optimizer = create_optimizer(config, model)
        optimizer = torch.optim.SGD(lr=0.0001,
                                    params=model.parameters(),
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
        # targets = torch.ones([len(images)], dtype=torch.long).to(device) * int(torch.argmax(original_output))
        # targets_list = torch.topk(original_output, k=3).indices.squeeze().detach()
        target_list = [i for i in range(10)]
        random.shuffle(target_list)
        targets_list = torch.Tensor(target_list).long().to(device)

        # find target candidates
        # targets_list = torch.Tensor([-1, -1])
        # targets_list[0] = int(torch.argmax(original_output))
        # adv_inputs = attack_model(images, targets)
        # adv_inputs.requires_grad = True
        # adv_outputs = model(adv_inputs.detach())
        # targets_list[1] = int(torch.argmax(adv_outputs))
        # targets_list = targets_list.long().to(device)
        # print('targets_list', targets_list)

        for _ in range(3):
            loss_list = torch.Tensor([0 for _ in range(10)])
            for i in range(10):
                # train_images, train_label = next(iter(train_loader))
                # images = merge_images(train_images, val_images, device)
                # outputs_list = []
                targets = targets_list[i % len(targets_list)].reshape([1])
                if int(targets) == int(original_class) or int(targets) == int(neighbour_class):
                    continue
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
                attack_model.set_mode_targeted_by_function(lambda image, label: targets)
                adv_inputs = attack_model(images, targets)
                adv_inputs.requires_grad = True
                adv_outputs = model(adv_inputs.detach())
                adv_loss = loss_func(adv_outputs, targets)

                # update target
                # outputs_list.append(outputs)
                # normal_output = model(images.detach())
                # normal_loss = loss_func(normal_output, targets)
                # loss_list[i] = torch.relu(adv_loss - normal_loss)  # untargeted
                # loss_list[i] = noise_loss - adv_loss  # targeted
                # loss_list[i] = adv_loss  # untargeted
                loss_list[i] = -1 * adv_loss  # targeted
                # print(int(targets.item()),
                #       '{:.4f}'.format(float(torch.relu(adv_loss - normal_loss))),
                #       '{:.4f}'.format(float(adv_loss)),
                #       outputs)
                # targets = torch.ones([len(images)], dtype=torch.long).to(device) * int(torch.argmax(outputs))
                # print(targets, torch.softmax(outputs, dim=1), torch.softmax(original_output, dim=1))

                # cur_original_output = model(images.detach())
                # cur_neighbour_output = model(neighbour_images.detach())
                # kl_loss_middle = nn.KLDivLoss(size_average=False, log_target=True)(
                #     torch.log_softmax(cur_original_output, dim=1),
                #     torch.log_softmax(cur_neighbour_output, dim=1)
                # )
                # kl_loss_ori = nn.KLDivLoss(size_average=False, log_target=True)(
                #     torch.log_softmax(cur_original_output, dim=1),
                #     torch.log_softmax(original_output.detach(), dim=1)
                # )
                # kl_loss_nei = nn.KLDivLoss(size_average=False, log_target=True)(
                #     torch.log_softmax(cur_neighbour_output, dim=1),
                #     torch.log_softmax(neighbour_output.detach(), dim=1)
                # )
                # loss_list[i] = kl_loss_middle + kl_loss_ori + kl_loss_nei
                # print('ori', cur_original_output)
                # print('nei', cur_neighbour_output)
                # print('{:.4f} {:.4f} {:.4f}'.format(float(kl_loss_middle), float(kl_loss_ori), float(kl_loss_nei)))

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
    # config = load_config(["train.batch_size", 1, "validation.batch_size", 1, "test.batch_size", 1])
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

    train_loader, test_loader = create_dataloader(config, is_train=True)
    train_loaders_by_class = create_dataloader_by_class(config)
    _, test_loss = create_loss(config)

    attack(config, model, train_loader, test_loader, train_loaders_by_class, test_loss, logger)


if __name__ == '__main__':
    main()
