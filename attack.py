#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from pytorch_image_classification.transforms import _get_dataset_stats
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)
from utils.debug_tools import clear_debug_image, save_image_stack

# attack parameters temporarily attached here
c = 2
lr = 0.1
momentum = 0.9
steps = 20
batch_size = 1


def load_config(options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if options:
        config.merge_from_list(options)
    update_config(config)
    config.freeze()
    return config


def cal_accuracy(output, target):
    with torch.no_grad():
        if torch.argmax(output) == torch.argmax(target):
            return 1
        return 0


class CWInfAttack(nn.Module):
    '''
    c:  coefficient of f value to be added to distance.
        Higher the c, higher the success rate and higher the distance
        see fig 2 of paper
    '''
    def __init__(self, model, config, c, lr, momentum, steps, device='cuda'):
        super(CWInfAttack, self).__init__()

        self.model = model
        self.config = config
        self.mean, self.std = _get_dataset_stats(config)
        self.c = c
        self.lr = lr
        self.steps = steps
        self.device = device
        self.momentum = momentum
        self.Normalize = torch_transforms.Normalize(
            self.mean, self.std
        )
        self.counter = 0

    def denormalize(self, images, is_tensor=True):
        if is_tensor:
            images = images.clone().detach().cpu().numpy()
        # image = np.squeeze(images)
        std = np.expand_dims(self.std, [0, 2, 3])
        mean = np.expand_dims(self.mean, [0, 2, 3])

        images = np.multiply(images, std)
        mean = np.multiply(np.ones_like(images), mean)
        images = images + mean
        # images = np.expand_dims(image, 0)
        if is_tensor:
            images = torch.Tensor(images).to(self.device)
        return images

    def forward(self, images, labels):
        # must be single image
        assert images.shape[0] == 1
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # image passed in are normalized, thus not in range [0,1]
        images = self.denormalize(images)

        w = self.get_init_w(images).detach()
        w.requires_grad = True
        images.requires_grad = False

        tau = 1

        best_adv_images = images.clone().detach()
        best_acc = 0
        best_delta = 1

        optimizer = torch.optim.SGD([w], lr=self.lr, momentum=self.momentum)
        # random target
        target = torch.randint(0, 9, labels.shape).to(self.device) + labels

        for step in range(self.steps):
            adv_images = self.w_to_adv_images(w)
            output = self.model(self.Normalize(adv_images))

            f_value = self.c * self.get_f_value(output, target)
            delta = self.w_to_delta(w, images)
            distance = self.inf_distance(delta, tau)
            loss = f_value + distance

            # update tau
            if torch.max(delta) < tau:
                tau = 0.9 * tau

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # print out results
            acc = cal_accuracy(output, target)
            avg_delta = torch.mean(delta)
            print('Acc: {}\tDelta: {}'.format(acc, avg_delta))
            if acc > best_acc:
                best_adv_images = adv_images
                best_acc = acc
                best_delta = avg_delta
            if acc == best_acc and avg_delta < best_delta:
                best_adv_images = adv_images
                best_acc = acc
                best_delta = avg_delta
            if acc == 1:
                break
        print('Batch finished: Acc: {}\tDelta: {}'.format(best_acc, best_delta))
        print('>>>>>')
        # pickle.dump(best_adv_images, open('adv_images_batch.pkl', 'wb'))
        if self.counter == 0:
            clear_debug_image()
        if self.counter < 10 and best_acc == 1:
            self.counter += 1
            save_image_stack(images, 'original input {} {}'.format(self.counter, best_delta))
            save_image_stack(best_adv_images, 'adversarial input {} {}'.format(self.counter, best_delta))
            # delta_image = torch.abs(best_adv_images - images)
            # print(torch.max(delta_image))
            # adjusted_delta = delta_image / torch.max(delta_image)
            # save_image_stack(adjusted_delta, 'adjusted delta')

        return best_adv_images, best_acc, best_delta

    @staticmethod
    def get_f_value(outputs, target):
        target_mask = torch.zeros_like(outputs)
        print(target_mask[:, target])
        target_mask[:, target] = 1
        src_p = torch.max(outputs * (1 - target_mask))
        target_p = torch.max(outputs * target_mask)
        f6 = torch.relu(src_p - target_p)
        return f6

    @staticmethod
    def inf_distance(delta, tau):
        dist_vec = torch.relu(delta - tau)
        return torch.sum(dist_vec)

    @staticmethod
    def w_to_adv_images(w):
        return 1/2 * (torch.tanh(w) + 1)

    @staticmethod
    def w_to_delta(w, x):
        return torch.abs(CWInfAttack.w_to_adv_images(w) - x)

    @staticmethod
    def get_init_w(x):
        return torch.atanh(2 * x - 1)


def attack(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()
    attack_model = CWInfAttack(model, config, c, lr, momentum, steps).cuda()

    accuracy_meter = AverageMeter()
    delta_meter = AverageMeter()
    adv_image_list = []

    for data, targets in tqdm.tqdm(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        adv_images, acc, delta = attack_model(data, targets)
        accuracy_meter.update(acc, 1)
        delta_meter.update(delta, 1)
        adv_image_list.append(adv_images)

    logger.info(f'Accuracy {accuracy_meter.avg:.4f} Delta {delta_meter.avg:.4f}')

    return adv_image_list, accuracy_meter.avg, delta_meter.avg


def main():
    config = load_config(["test.batch_size", 1])

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    # checkpointer = Checkpointer(model,
    #                             checkpoint_dir=output_dir,
    #                             # save_dir=output_dir,
    #                             logger=logger,
    #                             distributed_rank=get_rank())
    checkpointer = Checkpointer(model,
                                save_dir=output_dir)
    checkpointer.load(config.test.checkpoint)

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    attack(config, model, test_loader, test_loss, logger)




if __name__ == '__main__':
    main()
