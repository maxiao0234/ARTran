import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy
import utils
import random

import models
import read_HM


def get_args_parser():
    parser = argparse.ArgumentParser('ARTran training script', add_help=False)
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--fold', default=1, type=int)
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

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


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
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    count_positive = 0
    count_negative = 0
    positive_label = torch.ones(100).to(device)
    threshold = 50
    with torch.no_grad():
        for samples in data_loader_val:
            image = samples['image'].to(device, non_blocking=True)
            shift = samples['shift'].to(device, non_blocking=True)
            clsSEShift = samples['clsSEShift'][0].item()
            clsSEBaseline = samples['clsSEBaseline'][0].item()
            cube_id = samples['cube_id'][0].item()
            with torch.cuda.amp.autocast():
                outputs = model(image, shift=shift)

            pred_positive_shift = accuracy(outputs['post_clean_shift'], positive_label)[0]
            uncertainty_post = - outputs['post_clean_shift'] * torch.log(outputs['post_clean_shift'])
            uncertainty_post = torch.mean(torch.sum(uncertainty_post, dim=-1))
            if pred_positive_shift == 0:
                uncertainty_frame = 0
            else:
                uncertainty_frame = - (pred_positive_shift / 100) * torch.log(pred_positive_shift / 100)
                uncertainty_frame -= ((100 - pred_positive_shift) / 100) * torch.log((100 - pred_positive_shift) / 100)

            uncertainty_adjustment_list = []
            for shift_i in [0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3]:
                shift_uncertainty = torch.ones_like(shift).to(image.device) * shift_i
                outputs_uncertainty = model(image, shift=shift_uncertainty)
                pred_positive_clean_shift_uncertainty = outputs_uncertainty['post_clean_shift']
                uncertainty_adjustment_list.append(pred_positive_clean_shift_uncertainty)
            uncertainty_adjustment_list = torch.stack(uncertainty_adjustment_list, dim=0)
            uncertainty_adjustment = torch.var(uncertainty_adjustment_list, dim=0)
            uncertainty_adjustment = torch.mean(uncertainty_adjustment)

            if clsSEBaseline == 1:
                count_positive += 1
                if pred_positive_shift > threshold:
                    true_positive += 1
                    flag = 'TP'
                else:
                    false_negative += 1
                    flag = 'FN'
            else:
                count_negative += 1
                if pred_positive_shift > threshold:
                    false_positive += 1
                    flag = 'FP'
                else:
                    true_negative += 1
                    flag = 'TN'

            print(f'Flag: {flag}, ID: {cube_id}, Post: {uncertainty_post:.7f}, Adjustment: {uncertainty_adjustment:.7f}, Frame: {uncertainty_frame:>5}')

    accuracy_ = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision_ = true_positive / (true_positive + false_positive)
    recall_ = true_positive / (true_positive + false_negative)
    print(f'Accuracy: {accuracy_}')
    print(f'Precision: {precision_}')
    print(f'Recall: {recall_}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ARTran evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)