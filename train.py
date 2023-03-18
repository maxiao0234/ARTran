import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import math
import sys
from pathlib import Path
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy
import utils
import os
import random

import models


def get_args_parser():
    parser = argparse.ArgumentParser('ARTran training script', add_help=False)
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='ARTran', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--hw_shape', dest='cord', type=tuple)
    parser.add_argument('--kernel_size', dest='cord', type=tuple)
    parser.add_argument('--stride', dest='cord', type=tuple)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class CELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        output = torch.log(pred)
        return F.nll_loss(output, target)


def main(args):
    utils.init_distributed_mode(args)
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    dataset_train = read_HM.HMDataset(
        data_root=args.data_root,
        fold=args.fold,
        isTrain=True,
        adjustable=True,
        hw_shape=args.hw_shape,
    )
    dataset_val = read_HM.HMDataset(
        data_root=args.data_root,
        fold=args.fold,
        isTrain=True,
        hw_shape=args.hw_shape,
        shift=args.shift
    )

    if args.distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=100,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    model = models.AdjustableRobustTransformer(
        num_classes=args.nb_classes,
        hw_shape=args.hw_shape,
        in_chans=3,
        kernel_size=args.kernel_size,
        stride=args.stride
    )
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters}')

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    criterion = CELoss()
    max_accuracy = 0.0
    print_freq = 10

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        model.train()
        batch = 0
        for samples in data_loader_train:
            image = samples['image'].to(device, non_blocking=True)
            label = samples['clsSEBaseline'].to(device, non_blocking=True)
            label_shift = samples['clsSEShift'].to(device, non_blocking=True)
            shift = samples['shift'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(image, shift=shift)
                loss_class_b = criterion(outputs['post_noise_benchmark'], label)
                loss_class_s = criterion(outputs['post_noise_shift'], label_shift)
                loss_volume = outputs['volume']
                loss = (loss_class_b + loss_class_s) / 2 + loss_volume
            acc_benchmark = accuracy(outputs['post_noise_benchmark'], label)
            acc_shift = accuracy(outputs['post_noise_shift'], label_shift)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=is_second_order)
            torch.cuda.synchronize()

            if batch % print_freq == 0 and dist.get_rank() == 0:
                print(f'Epoch: {epoch}, Batch: {batch}/{len(data_loader_train)}, Acc: {acc_benchmark[0].item():.3f}%(Benchmark)/{acc_shift[0].item():.3f}%(Shift), Loss: {loss:.5f}, LR: {optimizer.param_groups[0]["lr"]:.5f}')
            batch += 1

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        model.eval()
        batch = 0
        total_acc = 0
        with torch.no_grad():
            for samples in data_loader_val:
                image = samples['image'].to(device, non_blocking=True)
                label = samples['clsSEBaseline'].to(device, non_blocking=True)
                label_shift = samples['clsSEShift'].to(device, non_blocking=True)
                shift = samples['shift'].to(device, non_blocking=True)
                cube_id = samples['cube_id'][0].item()

                with torch.cuda.amp.autocast():
                    outputs = model(image, shift=shift)
                    post_benchmark = outputs['post_clean_benchmark']
                    post_shift = outputs['post_clean_shift']

                acc_benchmark = accuracy(post_benchmark, label)
                acc_benchmark_cube = 100 if acc_benchmark[0].item() >= 50 else 0
                acc_shift = accuracy(post_shift, label_shift)
                acc_shift_cube = 100 if acc_shift[0].item() >= 50 else 0
                total_acc += acc_benchmark_cube

                if batch % print_freq == 0 and dist.get_rank() == 0:
                    print(
                        f'Test: {cube_id}, Batch: {batch}/{len(data_loader_train)}, Acc: {acc_benchmark_cube[0].item():.3f}%(Benchmark)/{acc_shift_cube[0].item():.3f}%(Shift)')
            batch += 1
        total_acc = total_acc / (len(data_loader_train) // 100)

        if max_accuracy < total_acc:
            max_accuracy = total_acc

        if dist.get_rank() == 0:
            print(f"Epoch: {epoch}, Accuracy on {len(dataset_val) // 100} test cubes: {total_acc:.2f}%")
            print(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if dist.get_rank() == 0:
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ARTran training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
