#!/usr/bin/env python
"""
   WaveGlow distillation script
   this is a modified version of train_waveglow.py that can be used to distill an existing waveglow model into
   a non-flow student (see models.students) by minimizing a mixture of MAE and STFT losses.
"""

import torch

import logging
import itertools

import torch.distributed as dist
import torch.utils.data as data
import torch.nn as nn
import torch

import random
import numpy as np
import argparse
import math

from utils.logger import Logger
from utils.logging_utils import setup_glog_stdout
from mel2samp import Mel2Samp, TRAIN_SPLIT_NAME, VAL_SPLIT_NAME
from utils.pprofiler import profiler as pp
from utils.distillation_loss import DistillationLoss

from utils.lr_schedules import OneCycleSchedule, MultiCycleSchedule, get_learning_rate
from models.waveglow_teacher import WaveGlowTeacher, DeterministicWaveGlowTeacher
from models.students import FlowStudent, WideFlowStudent, AffineStudent, WaveNetStudent
from models import defaults

from ya_tools import nirvana_tools

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

    rank = 0
    if args.local_rank is not None:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        rank = dist.get_rank()

    if rank == 0:
        nirvana_tools.copy_snapshot_to_out(out=args.logdir)
        logger = Logger(args.logdir, args.eval_interval)

    if args.local_rank is not None:
        # ensure that WaveGlowLogger.__init__(...) has finished
        dist.barrier()

    ckpt = Logger.load_last_checkpoint(args.logdir)  # FIXME: why logger?

    if args.deterministic_teacher:
        infer_teacher = DeterministicWaveGlowTeacher.load(args.teacher_path, train=False, device=device, fp16=True)
    else:
        infer_teacher = WaveGlowTeacher.load(args.teacher_path, train=False, device=device, fp16=True)

    if args.student_arch == 'flow':
        model = FlowStudent(in_channels=8, mel_channels=640, hid_channels=args.student_hid_channels,
                            n_wavenets=args.student_n_wavenets, wavenet_layers=args.student_wavenet_layers,
                            kernel_size=args.student_kernel_size)
    elif args.student_arch == 'wide_flow':
        model = WideFlowStudent(in_channels=8, mel_channels=640, hid_channels=args.student_hid_channels,
                                n_wavenets=args.student_n_wavenets, wavenet_layers=args.student_wavenet_layers,
                                kernel_size=args.student_kernel_size)
    elif args.student_arch == 'affine':
        model = AffineStudent(in_channels=8, mel_channels=640, hid_channels=args.student_hid_channels,
                                    n_wavenets=args.student_n_wavenets, wavenet_layers=args.student_wavenet_layers,
                                    kernel_size=args.student_kernel_size)
    elif args.student_arch == 'wavenet':
        model = WaveNetStudent(in_channels=8, mel_channels=640, hid_channels=args.student_hid_channels,
                                n_wavenets=args.student_n_wavenets, wavenet_layers=args.student_wavenet_layers,
                                kernel_size=args.student_kernel_size)

    if rank == 0:
        log.info("Teacher config:")
        for k, v in infer_teacher.config.items():
            log.info("\t{}: {}".format(k, v))
        log.info("Student:")
        log.info(repr(model))
        num_params = lambda model: sum((param.numel() for param in model.parameters()))
        log.info(f'Number of student parameters: {num_params(model)}')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.onecycle:
        log.info("Using OneCycleSchedule")
        optimizer = OneCycleSchedule(
            optimizer,
            learning_rate_base=args.onecycle_lr_base,
            warmup_steps=args.onecycle_warmup_steps,
            decay_rate=args.onecycle_decay_rate,
            learning_rate_min=args.onecycle_lr_min
        )
    elif args.multicycle:
        log.info("Using MultiCycleSchedule")
        optimizer = MultiCycleSchedule(
            optimizer,
            mode=args.multicycle_mode,
            step_size=args.multicycle_step_size,
            base_lr=args.multicycle_lr_base,
            max_lr=args.multicycle_lr_max,
            gamma=args.multicycle_gamma
        )

    if args.local_rank is not None:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    step = ckpt.get('step', 0)

    # TODO: override any optimizer settings?
    if ckpt['state_dict']:
        if hasattr(model, 'module'):
            model.module.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt['state_dict'])

        optimizer_state = ckpt.get("optimizer_state_dict", None)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        else:
            log.warning("No optimizer state found")
        log.info("warm start")

    if not (args.onecycle or args.multicycle):
        if get_learning_rate(optimizer) != args.learning_rate:
            log.warning("Updating optimizer learning rate ({}) as provided via --learning-rate: {}".format(
                get_learning_rate(optimizer), args.learning_rate
            ))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

    log.info("Training from step {}".format(step))

    train_dataset_kwargs = dict(
        split=TRAIN_SPLIT_NAME,
        segment_length=args.segment_length,
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax
    )
    if rank == 0:
        log.info("Train data config")
        for k, v in train_dataset_kwargs.items():
            log.info("\t{}: {}".format(k, v))

    dataset_train = Mel2Samp(
        args.dataset,
        **train_dataset_kwargs
    )

    validation_dataset_kwargs = dict(
        split=VAL_SPLIT_NAME,
        segment_length=None,
        filter_length=args.filter_length,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax
    )
    if rank == 0:
        log.info("Validation data config")
        for k, v in validation_dataset_kwargs.items():
            log.info("\t{}: {}".format(k, v))
    validation_dataset = Mel2Samp(args.dataset, **validation_dataset_kwargs)

    if args.local_rank is None:
        train_loader = data.DataLoader(
            dataset_train,
            num_workers=8,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=False,
            drop_last=True)
    else:
        sampler = data.distributed.DistributedSampler(dataset_train)
        train_loader = data.DataLoader(
            dataset_train,
            num_workers=8,
            shuffle=False,
            sampler=sampler,
            batch_size=args.batch_size,
            pin_memory=False,
            drop_last=True)

    validation_loader_kwargs = dict(
        num_workers=1, shuffle=False,
        batch_size=1, pin_memory=False
    )
    validation_loader = data.DataLoader(validation_dataset, **validation_loader_kwargs)

    distillation_loss = DistillationLoss(
        model, infer_teacher, infer_teacher=infer_teacher, teacher_dtype=torch.float16,
        stft_loss_coeff=args.stft_loss_coeff, mel=True, log=True,
        hop_length=args.hop_length, win_length=args.win_length, num_mels=args.n_mel_channels)

    model.train()
    stop = False
    batch_counter = 0
    epoch = step * args.virtual_batch_multiplier // len(train_loader)

    while not stop:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        if args.debug_single_sample_preload:
            sample = next(iter(train_loader))
            train_samples = itertools.repeat(sample, len(train_loader))
        else:
            train_samples = train_loader

        for mel, _unused_audio in train_samples:
            with pp("step"):
                if batch_counter == 0:
                    model.zero_grad()

                with pp("forward"):
                    input_audio = _unused_audio.to(device) if args.distill_on_real else None
                    loss, loss_mae, loss_stft = distillation_loss(mel.to(device), audio=input_audio, sigma=args.sigma)

                with pp("backward"):
                    loss.backward()
                    batch_counter += 1

                if batch_counter == args.virtual_batch_multiplier:
                    # here we have gradients
                    # but we didn't make a step yet
                    model_ = model if args.local_rank is None else model.module

                    grad_norm = total_norm_frobenius(model_.parameters())

                    loss_value = loss.item()
                    bad_loss = not math.isfinite(loss_value)
                    bad_grad = not math.isfinite(grad_norm)

                    if not bad_grad and args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    if args.safe_step and (bad_loss or bad_grad):
                        log.info("Skip. Loss {} Grad {}".format(loss_value, grad_norm))
                    else:
                        optimizer.step()
                        step += 1

                    batch_counter = 0

                    # average loss value for logs
                    if args.local_rank is not None:
                        dist.all_reduce(loss)
                        dist.all_reduce(loss_mae)
                        dist.all_reduce(loss_stft)
                        loss /= dist.get_world_size()
                        loss_mae /= dist.get_world_size()
                        loss_stft /= dist.get_world_size()

                    if rank == 0:
                        logger_step = args.log_step_offset + step
                        log.info("{:6d}: loss {:.4f} (mae {:.4f} ; stft {:.4f}) grad_l2 {:.3f}".format(
                            logger_step, loss.item(), loss_mae.item(), loss_stft.item(), grad_norm))
                        logger.add_scalar("loss/train/loss", loss.item(), logger_step)
                        logger.add_scalar("loss/train/loss_mae", loss_mae.item(), logger_step)
                        logger.add_scalar("loss/train/loss_stft", loss_stft.item(), logger_step)
                        logger.add_scalar("misc/grad_norm", grad_norm, logger_step)
                        logger.add_scalar("misc/learning_rate", get_learning_rate(optimizer), logger_step)

                    if step % args.eval_interval == 0:
                        model.train(False)
                        model_ = model if args.local_rank is None else model.module
                        val_loss_numerator = val_loss_mae_numerator = val_loss_stft_numerator = 0.
                        val_loss_denominator = 0  # total number of samples

                        with torch.no_grad(), pp("evaluate"):
                            for mel, _unused_audio in validation_loader:
                                input_audio = _unused_audio.to(device) if args.distill_on_real else None
                                loss, loss_mae, loss_stft = distillation_loss(mel.to(device), audio=input_audio, sigma=args.sigma)
                                val_loss_numerator += loss.item() * len(mel)
                                val_loss_mae_numerator += loss_mae.item() * len(mel)
                                val_loss_stft_numerator += loss_stft.item() * len(mel)
                                val_loss_denominator += len(mel)

                        if rank == 0:
                            logger_step = args.log_step_offset + step
                            val_loss = val_loss_numerator / val_loss_denominator
                            val_loss_mae = val_loss_mae_numerator / val_loss_denominator
                            val_loss_stft = val_loss_stft_numerator / val_loss_denominator
                            log.info("validation loss at {}: {:.4f}  (mae {:.4f} ; stft {:.4f})".format(
                                logger_step, val_loss, val_loss_mae, val_loss_stft))

                            logger.add_scalar('loss/val/loss', val_loss, logger_step)
                            logger.add_scalar('loss/val/loss_mae', val_loss_mae, logger_step)
                            logger.add_scalar('loss/val/loss_stft', val_loss_stft, logger_step)
                            logger.save_checkpoint(model_, optimizer, step) 
                        model.train()

                    if rank == 0 and step % args.train_info_interval == 0:
                        pp.print_report(printer=log.info)

                    if step >= args.num_steps:
                        stop = True
                        break
                else:
                    pass
        epoch += 1


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='JSON file for configuration')
    parser.add_argument('--dataset', type=str, required=True, help='directory with "train.txt" and "valid.txt" files')
    parser.add_argument('--logdir', type=str, required=True, help='directory for tensorboard logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    # train config
    parser.add_argument('--num-steps', default=1000, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument("--train-info-interval", default=100, type=int)
    parser.add_argument('--eval-interval', default=1000, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument("--grad-clip", default=None, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--clip-z-multiplier", default=None, type=float)
    parser.add_argument("--virtual-batch-multiplier", default=1, type=int)

    # distillation parameters
    parser.add_argument("--stft-loss-coeff", default=0.01, type=float)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument('--student-arch', default='wavenet', type=str, 
                                          choices=['flow', 'wide_flow', 'affine', 'wavenet'])
    parser.add_argument("--student_n_wavenets", default=4, type=int)
    parser.add_argument("--student_wavenet_layers", default=8, type=int)
    parser.add_argument("--student_hid_channels", default=96, type=int)
    parser.add_argument("--student_kernel_size", default=3, type=int)
    parser.add_argument("--deterministic_teacher", action='store_true')

    # mixed precision parameters
    parser.add_argument("--safe-step", action="store_true")

    # learning rate curve
    parser.add_argument("--onecycle", action="store_true")
    parser.add_argument("--onecycle-lr-base", type=float, default=1e-3)
    parser.add_argument("--onecycle-warmup-steps", type=int, default=10000)
    parser.add_argument("--onecycle-decay-rate", type=float, default=0.2)
    parser.add_argument("--onecycle-lr-min", type=float, default=1e-5)

    parser.add_argument("--multicycle", action="store_true")
    parser.add_argument("--multicycle-mode", type=str, choices=["tri", "tri2", "exp_range"])
    parser.add_argument("--multicycle-step-size", type=int, default=80000)
    parser.add_argument("--multicycle-lr-base", type=float, default=1e-5)
    parser.add_argument("--multicycle-lr-max", type=float, default=1e-3)
    parser.add_argument("--multicycle-gamma", type=float, default=0.9)

    parser.add_argument("--log-step-offset", type=int, default=0)

    # data config
    parser.add_argument('--segment-length', default=defaults.SEGMENT_LENGTH, type=int)
    parser.add_argument('--sampling-rate', default=defaults.SAMPLING_RATE, type=int)
    parser.add_argument('--filter-length', default=defaults.STFT_FILTER_LENGTH, type=int)
    parser.add_argument('--hop-length', default=defaults.STFT_HOP_LENGTH, type=int)
    parser.add_argument('--win-length', default=defaults.STFT_WIN_LENGTH, type=int)
    parser.add_argument('--mel-fmin', default=defaults.MEL_FMIN, type=float)
    parser.add_argument('--mel-fmax', default=defaults.MEL_FMAX, type=float)

    # model config
    parser.add_argument('--n-mel-channels', default=defaults.MEL_CHANNELS, type=int)
    parser.add_argument('--n-flows', default=defaults.WG_N_FLOWS, type=int)
    parser.add_argument('--n-group', default=defaults.WG_N_GROUP, type=int)
    parser.add_argument('--n-early-every', default=defaults.WG_N_EARLY_EVERY, type=int)
    parser.add_argument('--n-early-size', default=defaults.WG_N_EARLY_SIZE, type=int)
    parser.add_argument('--wavenet-n-layers', default=defaults.WAVENET_N_LAYERS, type=int)
    parser.add_argument('--wavenet-n-channels', default=defaults.WAVENET_N_CHANNELS, type=int)
    parser.add_argument('--wavenet-kernel-size', default=defaults.WAVENET_KERNEL_SIZE, type=int)
    parser.add_argument('--upsample-multistage', action="store_true")
    parser.add_argument("--decompose-convinv", action="store_true")
    parser.add_argument("--wavenet-causal-layers", type=int, nargs="+")

    # distributed-specific parameter
    parser.add_argument('--local_rank', type=int, default=None)

    train(parser.parse_args())
