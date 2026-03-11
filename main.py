import os
import json
import argparse
import time 
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np

from get_dataset import get_dataset
from models.cgm_eeg import CGM
from models.biot import BIOTClassifier
import optim
from engine import train_epoch, evaluate
import utils

def get_args():
    parser = argparse.ArgumentParser('CGM-EEG Training Script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='min lr for cyclic schedulers that hit 0 (default: 1e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of LR warmup steps, will overload warmup_epochs if set > 0')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPT',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPS',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--model', default='cgm', type=str, metavar='MODEL')
    parser.add_argument('--emb_size', default=256, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--heads', default=4, type=int,
                        help='Number of heads for Transformer-based models')
    parser.add_argument('--n_channels', default=16, type=int)
    parser.add_argument('--n_classes', default=1, type=int)
    parser.add_argument('--dataset', default='tuev', type=str, help="Dataset to train: CHBMIT|TUEV|TUSZ")
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--resume', default='')
    
    return parser.parse_args()

def get_model(args):
    if args.model == 'cgm':
        model = CGM(
            emb_size=args.emb_size,
            depth=args.depth//2,
            n_channels=args.n_channels, 
            n_classes=args.n_classes,
            n_fft=args.sfreq, 
            hop_length=args.sfreq // 2,
        )
    elif args.model == 'biot':
        model = BIOTClassifier(
            n_classes=args.n_classes,
            n_fft=args.sfreq,
            hop_length=args.sfreq // 2,
        )
    else:
        raise ValueError(f"Unknow model: {args.model}")
    return model

def main(args):
    print(args)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    train_dataset, val_dataset, test_dataset, args = get_dataset(args, verbose=True)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, 
                                  num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, 
                                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, 
                                 num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
    model = get_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params}")

    niter_per_epoch = len(train_dataset) // args.batch_size
    print(f"LR = {args.lr:.6f}")
    print(f"Batch size = {args.batch_size}")
    print(f"Number of training examples = {len(train_dataset)}") 
    print(f"Number of training training per epoch = {niter_per_epoch}")

    optimizer = optim.create_optimizer(args, model)
    loss_scaler = optim.NativeScalerWithGradNormCount()

    lr_schedule_values = optim.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, niter_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = optim.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, niter_per_epoch
    )

    if args.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"Criterion = {str(criterion)}")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = float('inf')
    max_val = 0.0
    max_test = 0.0
    patience = 0
    if args.n_classes == 1:
        metric = 'auroc'
    else:
        metric = 'kappa'

    for epoch in range(args.epochs):
        start_steps = epoch * niter_per_epoch
        train_stats = train_epoch(train_dataloader, model, criterion, optimizer, device, epoch, loss_scaler, args.clip_grad, 
                                  start_steps, lr_schedule_values, wd_schedule_values, niter_per_epoch, args.n_classes)
        val_stats = evaluate(val_dataloader, model, device, header='Val:', n_classes=args.n_classes)
        print("Val:  " + " | ".join(f"{k}: {v:.4f}" for k, v in val_stats.items()))
        test_stats = evaluate(test_dataloader, model, device, header='Test:', n_classes=args.n_classes)
        print("Test: " + " | ".join(f"{k}: {v:.4f}" for k, v in test_stats.items()))
        epoch_name = f"{epoch}-{metric}{test_stats[metric]:.4f}"
        if val_stats[metric] > max_val:
            patience = 0
            min_loss = val_stats['loss']
            max_val = val_stats[metric]
            max_test = test_stats[metric]
            print(f"New best model found at epoch {epoch}, saving model...")
            utils.save_model(args, epoch_name, model, optimizer, loss_scaler)
        else: 
            patience += 1
            if patience == args.patience: 
                print(f"Early stopping at epoch {epoch}, patience={args.patience} reached.")
                print(f'Min val loss: {min_loss:.2f}, Max, val {metric}: {max_val:.2f}, Max test {metric}: {max_test:.2f}')
                break
    
        print(f'Min val loss: {min_loss:.2f}, Max, val {metric}: {max_val:.2f}, Max test {metric}: {max_test:.2f}')
        
        log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                     **{f'val_{k}': float(v) for k, v in val_stats.items()},
                     **{f'test_{k}': float(v) for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_params}
        if args.output_dir:
            subdir_name = f"{args.dataset}_{args.model}_{args.seed}"
            log_dir = os.path.join(args.output_dir, subdir_name)
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}') 

if __name__ == '__main__':
    args = get_args()
    main(args)

