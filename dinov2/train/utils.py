import os
import math
import torch

import dinov2.distributed as distributed


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

def update_schedules(optimizer, schedulers, train_step):
    lr = schedulers['lr'][train_step]
    wd = schedulers['wd'][train_step]
    momentum = schedulers['momentum'][train_step]
    teacher_temp = schedulers['teacher_temp'][train_step]
    last_layer_lr = schedulers['last_layer_lr'][train_step]
    apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

    return momentum, teacher_temp

def apply_gradient_operations(cfg, model, optimizer, fp16_scaler):
    if fp16_scaler is not None:
        fp16_scaler.unscale_(optimizer)

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.div_(cfg.train.grad_accum_steps)

    if cfg.optim.clip_grad:
        for v in model.student.values():
            v.clip_grad_norm_(cfg.optim.clip_grad)

    if fp16_scaler is not None:
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
    else:
        optimizer.step()
        
def log_training_step(metric_logger, loss_dict, schedulers, train_step, current_batch_size):
    if distributed.get_global_size() > 1:
        for v in loss_dict.values():
            torch.distributed.all_reduce(v)
    loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

    if math.isnan(sum(loss_dict_reduced.values())):
        logger.error(f"NaN detected in reduced loss at iteration {iteration}")
        logger.info(f"Reduced loss dict: {loss_dict_reduced}")
        raise AssertionError

    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    metric_logger.update(lr=schedulers['lr'][train_step])
    metric_logger.update(wd=schedulers['wd'][train_step])
    metric_logger.update(mom=schedulers['momentum'][train_step])
    metric_logger.update(last_layer_lr=schedulers['last_layer_lr'][train_step])
    metric_logger.update(current_batch_size=current_batch_size)
    metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

def do_test(cfg, model, iteration):
    torch.cuda.synchronize()
    
    new_state_dict = {k: v.cpu() for k, v in model.teacher.state_dict().items()}

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        
        torch.cuda.empty_cache()
        
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
