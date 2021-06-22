#!/usr/bin/env python
"""
   SRFlow distillation script
"""

import logging

import torch.utils.data as data
import torch.nn as nn
import torch

import math
import random
import numpy as np
import argparse

from utils.logger import Logger
from utils.logging_utils import setup_glog_stdout
from utils.pprofiler import profiler as pp

from utils.util import num_params, opt_get, load_teacher
from utils.eval_utils import compute_metrics, compute_metrics_on_train_patches, \
                             compute_validation_metrics, format_measurements, get_sr, get_sr_with_epses
from models.modules import Student_arch

from torchvision import transforms
from datasets import DF2K
import lpips

log = setup_glog_stdout(logging.getLogger())
log.setLevel(logging.INFO)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


@pp("grad norm computation")
def total_norm_frobenius(parameters):
    # compute total Frobenius norm of parameters
    total_norm = 0
    for p in filter(lambda p: p.grad is not None, parameters):
        param_norm = p.grad.data.norm()
        total_norm += param_norm.item() ** 2.0
    total_norm = total_norm ** (1. / 2.0)
    return total_norm


def train(args):
    torch.backends.cudnn.enabled = True         # default
    torch.backends.cudnn.benchmark = True       # non-default
    torch.backends.cudnn.deterministic = False  # default

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = Logger(args.logdir, args.eval_interval)

    #######################
    # Load SRFlow Teacher #
    #######################

    teacher_model, opt = load_teacher(args.config, dtype=torch.float32, device=device)
    
    opt['GT_size'] = args.patch_size
    opt['train_dtype'] = torch.float32 # fp16 and mixed precision lead to Inf/Nan. 
    opt['eval_dtype'] = torch.float32 # fp16 does not speed up the teacher significantly. 

    # One needs to make a single forward pass to prepare the teacher model
    dummy_input = torch.randn(1, 3, 16, 16).to(device, opt['eval_dtype'])
    get_sr(teacher_model, opt, dummy_input)
    teacher_model = nn.DataParallel(teacher_model).train(False)
    
    #########################
    # Create SRFlow Student #
    #########################

    if opt['scale'] == 4:
        from models.modules.RRDBNet_arch import SmallRRDB3C, RRDB 
        student_model = Student_arch.StudentX4(rrdb_block=RRDB, flow_block=SmallRRDB3C, K=6, opt=opt)
    elif opt['scale'] == 8:
        from models.modules.RRDBNet_arch import ResidualDenseBlock_3C, RRDB
        student_model = Student_arch.StudentX8(rrdb_block=ResidualDenseBlock_3C, flow_block=RRDB, K=1, opt=opt)
    else:
        raise Exception("Wrong scale: this code supports only x4 and x8 scaling factors")

    if args.use_pretrained_lr_encoder:
        lr_encoder_path = opt_get(opt, ['lr_encoder_path'], None)
        student_model.RRDB.load_state_dict(torch.load(lr_encoder_path))

    log.info("Student:")
    log.info(repr(student_model))
    log.info(f'Number of student parameters: {num_params(student_model)}')

    student_model = nn.DataParallel(student_model).to(device, opt['train_dtype'])
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epoch_size, gamma=args.gamma)
    
    ###############################
    # Loading the last checkpoint #
    ###############################

    ckpt = Logger.load_last_checkpoint(args.logdir)
    start_epoch = ckpt.get('epoch', 0)
    log.info("Training from epoch {}".format(start_epoch + 1))
    if ckpt['state_dict']:
        if hasattr(student_model, 'module'):
            student_model.module.load_state_dict(ckpt['state_dict'])
        else:
            student_model.load_state_dict(ckpt['state_dict'])

        optimizer_state = ckpt.get("optimizer_state_dict", None)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        else:
            log.warning("No optimizer state found")

        scheduler_state = ckpt.get("scheduler_state_dict", None)
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        else:
            log.warning("No scheduler_state found")

        log.info("warm start")

    ############
    # Datasets #
    ############

    train_dataset = DF2K(args.dataset, split='train', scale=opt['scale'], 
                         GT_size=args.patch_size, totensor=transforms.ToTensor())
    assert len(train_dataset) == 139579  # TODO: remove

    train_loader = data.DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=False,
        drop_last=True)

    validation_dataset = DF2K(args.dataset, split='val', scale=opt['scale'])

    ############
    # Training #
    ############

    student_model.train()
    metrics = ['PSNR', 'SSIM', 'LPIPS', 'LRC PSNR']
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device, opt['train_dtype'])
        
    for epoch in range(start_epoch + 1, args.num_epochs + 1, 1):
        log.info(f'Epoch {epoch} of {args.num_epochs}')
    
        for lrs, gts in train_loader:
            with pp("step"):
                # Sample from the teacher
                with pp("forward_teacher"):
                    targets, epses = get_sr_with_epses(teacher_model, opt, lq=lrs)

                    # Reject teacher samples with large out-of-range values 
                    counter = 0
                    while (targets > 5).any():
                        if counter == 5: break
                        targets, epses = get_sr_with_epses(teacher_model, opt, lq=lrs)
                        counter += 1
                    # if the teacher always produces samples with large out-of-range values, skip this batch
                    # Note that this almost never happens
                    if counter == 5:
                        continue

                # Training step
                with pp("forward_student"):
                    targets = targets.to(opt['train_dtype'])
                    epses = [eps.to(opt['train_dtype']) for eps in epses]
                    preds = student_model(lr=lrs.to(device, opt['train_dtype']), epses=epses)
                    mae = abs(preds - targets).mean()
                    lpips_loss = lpips_loss_fn.forward(2 * preds - 1, 2 * targets.clamp(0, 1) - 1).mean()
                    loss = mae + args.alpha * lpips_loss

                with pp("backward"):    
                    loss.backward()
                
                grad_norm = total_norm_frobenius(student_model.parameters())
                loss_value = loss.item()
                bad_loss = not math.isfinite(loss_value)
                bad_grad = not math.isfinite(grad_norm) 

                if not bad_grad and args.grad_clip is not None:
                    nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)

                if bad_loss or bad_grad:
                    log.info("Skip. Loss {} Grad {}".format(loss_value, grad_norm))
                else:
                    optimizer.step()

                optimizer.zero_grad()
                
                # Logging 
                step = optimizer._step_count
                logger.add_scalar("train/loss", loss.item(), step)
                logger.add_scalar("train/mae", mae.item(), step)
                logger.add_scalar("train/lpips_loss", lpips_loss.item(), step)
                logger.add_scalar("train/grad_norm", grad_norm, step)
                log.info("training loss at {}: {:.4f}  (mae {:.4f} ; lpips {:.4f})".format(
                        step, loss.item(), mae.item(), lpips_loss.item()))

                # Evaluate metrics
                if step % args.eval_interval == 0:
                    with pp("evaluate"):
                        student_model.train(False)
                        student_model.to(opt['eval_dtype'])

                        # Compute metrics on the train patches
                        meas = compute_metrics_on_train_patches(student_model, opt, lrs[:32], gts[:32])
                        str_out = format_measurements(meas)
                        log.info(f'metrics on train patches at {step}: ' + str_out)
                        for key in metrics:    
                            logger.add_scalar('train/' + key, meas[key], step)

                        # Compute metrics on full-sized validation images
                        meas = compute_validation_metrics(student_model, opt, validation_dataset)
                        str_out = format_measurements(meas)
                        log.info(f'validation metrics at {step}: ' + str_out)
                        for key in metrics:    
                            logger.add_scalar('val/' + key, meas[key], step)

                        student_model.to(opt['train_dtype'])
                        student_model.train(True)

            if step % args.train_info_interval == 0:
                pp.print_report(printer=log.info)

        scheduler.step()
        log.info(f'Save checkpoint after {epoch} epochs')
        logger.save_checkpoint(student_model, optimizer, scheduler, epoch) 

    # Final student evaluation
    student_model.train(False)
    student_model.to(opt['eval_dtype'])
    meas = compute_metrics(student_model, opt, opt['dataroot_LR'], opt['dataroot_GT'], test_dir=args.logdir)


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YML file for configuration')
    parser.add_argument('--dataset', type=str, required=True, help='directory with TODO files')
    parser.add_argument('--logdir', type=str, required=True, help='directory for tensorboard logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    # train config
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument('--lr-epoch-size', default=25, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)

    parser.add_argument("--train-info-interval", default=500, type=int)
    parser.add_argument('--eval-interval', default=500, type=int)
    parser.add_argument('--ckpt-interval', default=20, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument("--grad-clip", default=100., type=float)

    parser.add_argument('--patch-size', default=128, type=int)

    # distillation parameters
    parser.add_argument("--alpha", default=10.0, type=float, help='coefficient in the distillation loss')
    parser.add_argument("--K", default=6, type=int, help='number of RRDB modules at each level')
    parser.add_argument("--use-pretrained-lr-encoder", action="store_true")

    train(parser.parse_args())
