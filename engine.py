# Code adapted from the BIOT repository:
# https://github.com/soupeeli/NeuroBOLT

import sys
import math
import torch
import torch.nn as nn
import numpy as np
import utils

def train_epoch(dataloader, model, criterion, optimizer, device, epoch, loss_scaler, 
                max_norm, start_steps, lr_schedule_values, wd_schedule_values, 
                niter_per_epoch, n_classes=1):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    model.train()
 
    optimizer.zero_grad()
    
    for iter_step, batch in enumerate(metric_logger.log_every(dataloader, 100, 'Epoch: [{}]'.format(epoch))):
        if iter_step >= niter_per_epoch:
            continue
        it = start_steps + iter_step  # global training iteration
        if (lr_schedule_values is not None or wd_schedule_values is not None):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        inputs = batch[0].to(device, non_blocking=True) 
        targets = batch[-1].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(inputs)
            if n_classes == 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, targets.float())
            if n_classes == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()

        probs = probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        metrics = utils.get_metrics(probs, preds, targets, n_classes)
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(bacc=metrics[1])
        if n_classes == 1:
            metric_logger.update(auroc=metrics[2])
            metric_logger.update(precision=metrics[3])
            metric_logger.update(recall=metrics[4])
        else:
            metric_logger.update(f1=metrics[2])
            metric_logger.update(kappa=metrics[3])
        #metric_logger.update(loss_scale=loss_scale_value)

        lr_value = 0.
        for group in optimizer.param_groups:
            lr_value = max(lr_value, group["lr"])
        metric_logger.update(lr=lr_value)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(dataloader, model, device, header='Test:', n_classes=1):
    if n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    for step, batch in enumerate(metric_logger.log_every(dataloader, 50, header)):
        inputs = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(inputs)
            if n_classes == 1:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets.float())
            if n_classes == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)

        probs = probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(targets)

        metric_logger.update(loss=loss.item())
        
    metric_logger.synchronize_between_processes()
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = utils.get_metrics(all_probs, all_preds, all_targets, n_classes)
    
    if n_classes == 1:
        stats = {"acc": metrics[0], "bacc": metrics[1], "auroc": metrics[2], "precision": metrics[3], "recall": metrics[4], "loss": metric_logger.loss.global_avg}
    else:
        stats = {"acc": metrics[0], "bacc": metrics[1], "f1": metrics[2], "kappa": metrics[3], "loss": metric_logger.loss.global_avg}

    return stats