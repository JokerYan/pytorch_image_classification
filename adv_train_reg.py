#!/usr/bin/env python

import argparse
import pathlib
import time

from custom_loss.llv_loss import LocalLipschitzValueLoss
from pytorch_image_classification.models import create_input_sigmoid_model

try:
    import apex
except ImportError:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
from torchattacks import PGD

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_image_classification.config.config_node import ConfigNode
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)

global_step = 0

# adv training hyper-parameters
epsilon = 8 / 255
alpha = 10 / 255
attack_target_class = None  # -1 when random targeted


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    if args.target is not None:
        global attack_target_class
        attack_target_class = args.target
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config


def send_targets_to_device(config, targets, device):
    if config.augmentation.use_mixup or config.augmentation.use_cutmix:
        t1, t2, lam = targets
        targets = (t1.to(device), t2.to(device), lam)
    elif config.augmentation.use_ricap:
        labels, weights = targets
        labels = [label.to(device) for label in labels]
        targets = (labels, weights)
    else:
        targets = targets.to(device)
    return targets


def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader,
          logger, tensorboard_writer, tensorboard_writer2):
    global global_step

    logger.info(f'Train {epoch} {global_step}')

    device = torch.device(config.device)

    model.train()
    criterion_kl = nn.KLDivLoss(size_average=False)

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        step += 1
        global_step += 1
        batch_size = len(data)

        if get_rank() == 0 and step == 1:
            if config.tensorboard.train_images:
                image = torchvision.utils.make_grid(data,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Train/Image', image, epoch)

        data = data.to(device,
                       non_blocking=config.train.dataloader.non_blocking)
        targets = send_targets_to_device(config, targets, device)

        # generate fgsm adv examples
        normal_outputs = model(data)
        delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
        noise_inputs = data.detach() + delta
        noise_inputs.requires_grad = True
        noise_outputs = model(noise_inputs)

        assert attack_target_class is None  # only support un-targeted attacks
        # un-targeted attack
        # loss = loss_func(noise_outputs, targets)  # loss to be maximized
        loss = (1.0 / batch_size) * criterion_kl(torch.log_softmax(noise_outputs, dim=1),
                                                        torch.softmax(normal_outputs, dim=1))
        input_grad = torch.autograd.grad(loss, noise_inputs)[0]
        delta = delta + alpha * torch.sign(input_grad)
        delta.clamp_(-epsilon, epsilon)

        optimizer.zero_grad()

        adv_inputs = data.detach() + delta.detach()
        adv_outputs = model(adv_inputs)

        data.requires_grad = False
        noise_inputs.requires_grad = False
        adv_inputs.requires_grad = False

        # natural_loss = loss_func(normal_outputs, targets)
        # robust_loss = (1.0 / batch_size) * criterion_kl(torch.log_softmax(adv_outputs, dim=1),
        #                                                 torch.softmax(normal_outputs, dim=1))
        # loss = natural_loss + robust_loss
        loss = loss_func(adv_outputs, targets)
        loss.backward()
        optimizer.step()

        acc1, acc5 = compute_accuracy(config,
                                      adv_outputs,
                                      targets,
                                      augmentation=True,
                                      topk=(1, 5))

        if config.train.distributed:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc1_all_reduce = dist.all_reduce(acc1,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc5_all_reduce = dist.all_reduce(acc5,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            loss_all_reduce.wait()
            acc1_all_reduce.wait()
            acc5_all_reduce.wait()
            loss.div_(dist.get_world_size())
            acc1.div_(dist.get_world_size())
            acc5.div_(dist.get_world_size())

        loss = loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()

        num = data.size(0)
        loss_meter.update(loss, num)
        acc1_meter.update(acc1, num)
        acc5_meter.update(acc5, num)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if get_rank() == 0:
            if step % config.train.log_period == 0 or step == len(
                    train_loader):
                logger.info(
                    f'Epoch {epoch} '
                    f'Step {step}/{len(train_loader)} '
                    f'lr {scheduler.get_last_lr()[0]:.6f} '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                    f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

                tensorboard_writer2.add_scalar('Train/RunningLoss',
                                               loss_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc1',
                                               acc1_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc5',
                                               acc5_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningLearningRate',
                                               scheduler.get_last_lr()[0],
                                               global_step)

        scheduler.step()

    if get_rank() == 0:
        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

        tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc5', acc5_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
        tensorboard_writer.add_scalar('Train/LearningRate',
                                      scheduler.get_last_lr()[0], epoch)


def validate(epoch, config, model, loss_func, val_loader, logger,
             tensorboard_writer):
    logger.info(f'Val {epoch}')

    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(val_loader):
            if get_rank() == 0:
                if config.tensorboard.val_images:
                    if epoch == 0 and step == 0:
                        image = torchvision.utils.make_grid(data,
                                                            normalize=True,
                                                            scale_each=True)
                        tensorboard_writer.add_image('Val/Image', image, epoch)

            data = data.to(
                device, non_blocking=config.validation.dataloader.non_blocking)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            acc1, acc5 = compute_accuracy(config,
                                          outputs,
                                          targets,
                                          augmentation=False,
                                          topk=(1, 5))

            if config.train.distributed:
                loss_all_reduce = dist.all_reduce(loss,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                acc1_all_reduce = dist.all_reduce(acc1,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                acc5_all_reduce = dist.all_reduce(acc5,
                                                  op=dist.ReduceOp.SUM,
                                                  async_op=True)
                loss_all_reduce.wait()
                acc1_all_reduce.wait()
                acc5_all_reduce.wait()
                loss.div_(dist.get_world_size())
                acc1.div_(dist.get_world_size())
                acc5.div_(dist.get_world_size())
            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        logger.info(f'Epoch {epoch} '
                    f'loss {loss_meter.avg:.4f} '
                    f'acc@1 {acc1_meter.avg:.4f} '
                    f'acc@5 {acc5_meter.avg:.4f}')

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')

    if get_rank() == 0:
        if epoch > 0:
            tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc1', acc1_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc5', acc5_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
        if config.tensorboard.model_params:
            for name, param in model.named_parameters():
                tensorboard_writer.add_histogram(name, param, epoch)


def random_target_function(images, labels):
    attack_target = torch.remainder(torch.randint(1, 9, labels.shape).cuda() + labels, 10)
    # attack_target = torch.ones_like(labels).cuda() * 9
    return attack_target


def pgd_validate(epoch, config, model, loss_func, val_loader, logger,
             tensorboard_writer):
    logger.info(f'PGD Val {epoch}')

    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    start = time.time()

    counter = 0
    for step, (data, targets) in enumerate(val_loader):
        counter += 1
        if counter > 10:
            break
        if get_rank() == 0:
            if config.tensorboard.val_images:
                if epoch == 0 and step == 0:
                    image = torchvision.utils.make_grid(data,
                                                        normalize=True,
                                                        scale_each=True)
                    tensorboard_writer.add_image('PGD Val/Image', image, epoch)

        data = data.to(
            device, non_blocking=config.validation.dataloader.non_blocking)
        targets = targets.to(device)

        # generate pgd adv inputs
        pgd_model = PGD(model, eps=8/255, alpha=2/255, steps=5)
        if attack_target_class is not None and attack_target_class != -1:
            attack_target_tensor = torch.ones_like(targets) * attack_target_class
            # avoid target == attack target
            attack_target_tensor[targets == attack_target_tensor] = (attack_target_class + 1) % config.dataset.n_classes
            pgd_model.set_mode_targeted_by_function(lambda images, labels: attack_target_tensor)  # targeted attack
        elif attack_target_class == -1:  # random targeted
            pgd_model.set_mode_targeted_by_function(random_target_function)

        adv_inputs = pgd_model(data, targets)

        outputs = model(adv_inputs)
        loss = loss_func(outputs, targets)

        acc1, acc5 = compute_accuracy(config,
                                      outputs,
                                      targets,
                                      augmentation=False,
                                      topk=(1, 5))

        if config.train.distributed:
            loss_all_reduce = dist.all_reduce(loss,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc1_all_reduce = dist.all_reduce(acc1,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            acc5_all_reduce = dist.all_reduce(acc5,
                                              op=dist.ReduceOp.SUM,
                                              async_op=True)
            loss_all_reduce.wait()
            acc1_all_reduce.wait()
            acc5_all_reduce.wait()
            loss.div_(dist.get_world_size())
            acc1.div_(dist.get_world_size())
            acc5.div_(dist.get_world_size())
        loss = loss.item()
        acc1 = acc1.item()
        acc5 = acc5.item()

        num = data.size(0)
        loss_meter.update(loss, num)
        acc1_meter.update(acc1, num)
        acc5_meter.update(acc5, num)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'acc@1 {acc1_meter.avg:.4f} '
                f'acc@5 {acc5_meter.avg:.4f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if get_rank() == 0:
        if epoch > 0:
            tensorboard_writer.add_scalar('PGD Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('PGD Val/Acc1', acc1_meter.avg, epoch)
        tensorboard_writer.add_scalar('PGD Val/Acc5', acc5_meter.avg, epoch)
        tensorboard_writer.add_scalar('PGD Val/Time', elapsed, epoch)
        if config.tensorboard.model_params:
            for name, param in model.named_parameters():
                tensorboard_writer.add_histogram(name, param, epoch)


def main():
    global global_step

    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)

    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)

    output_dir = pathlib.Path(config.train.output_dir)
    if get_rank() == 0:
        if not config.train.resume and output_dir.exists():
            raise RuntimeError(
                f'Output directory `{output_dir.as_posix()}` already exists')
        output_dir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            if diff is not None:
                save_config(diff, output_dir / 'config_min.yaml')

    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    train_loader, val_loader = create_dataloader(config, is_train=True)

    if config.custom_model.name:
        model = create_input_sigmoid_model(config)
    else:
        model = create_model(config)
    macs, n_params = count_op(config, model)
    logger.info(f'MACs   : {macs}')
    logger.info(f'#params: {n_params}')

    optimizer = create_optimizer(config, model)
    if config.device != 'cpu' and config.train.use_apex:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=config.train.precision)
    model = apply_data_parallel_wrapper(config, model)

    scheduler = create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)

    start_epoch = config.train.start_epoch
    scheduler.last_epoch = start_epoch
    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()
    elif config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(model,
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
            config, output_dir, purge_step=config.train.start_epoch + 1)
        tensorboard_writer2 = create_tensorboard_writer(
            config, output_dir / 'running', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()

    train_loss, val_loss = create_loss(config)

    if (config.train.val_period > 0 and start_epoch == 0
            and config.train.val_first):
        validate(0, config, model, val_loss, val_loader, logger,
                 tensorboard_writer)

    for epoch, seed in enumerate(epoch_seeds[start_epoch:], start_epoch):
        epoch += 1

        np.random.seed(seed)
        train(epoch, config, model, optimizer, scheduler, train_loss,
              train_loader, logger, tensorboard_writer, tensorboard_writer2)

        if config.train.val_period > 0 and (epoch % config.train.val_period
                                            == 0):
            validate(epoch, config, model, val_loss, val_loader, logger,
                     tensorboard_writer)
            pgd_validate(epoch, config, model, val_loss, val_loader, logger,
                     tensorboard_writer)

        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        if (epoch % config.train.checkpoint_period
                == 0) or (epoch == config.scheduler.epochs):
            checkpoint_config = {
                'epoch': epoch,
                'global_step': global_step,
                'config': config.as_dict(),
            }
            checkpointer.save(f'checkpoint_{epoch:05d}', **checkpoint_config)

    tensorboard_writer.close()
    tensorboard_writer2.close()


if __name__ == '__main__':
    main()
