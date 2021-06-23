from collections import OrderedDict

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.measure import Measure, psnr
from utils.imresize import imresize
from utils.util import patchify, fiFindByWildcard, t, rgb, imread, imwrite, impad
from models.modules.flow import GaussianDiag


def get_epses(opt, batch_size, lr_shape, eps_std, seed=None):
    if seed: torch.manual_seed(seed)
    C = opt['flowUpsamplerNet']['C']
    H = int(opt['scale'] * lr_shape[2] // opt['flowUpsamplerNet']['scaleH'])
    W = int(opt['scale'] * lr_shape[3] // opt['flowUpsamplerNet']['scaleW'])

    z = GaussianDiag.sample_eps([batch_size, C, H, W], eps_std)
    epses = [z]
    levels = int(np.log2(opt['scale']) + 1)
    for level in range(2, levels):
        new_C = 2 ** (level-2) * C // 4**level
        new_H = H * 2**level
        new_W = W * 2**level
        eps = GaussianDiag.sample_eps([batch_size, new_C, new_H, new_W], eps_std)
        epses.append(eps)
    return epses[::-1]


@torch.no_grad()
def get_sr_with_epses(model, opt, lq, heat=None, epses=None):
    assert not model.training 
    if epses is None:
        heat = opt['heat'] if heat is None else heat
        epses = get_epses(opt, lq.shape[0], lq.shape, heat)

    lr = lq.to(opt['device'], opt['eval_dtype'])
    epses = [eps.to(opt['device'], opt['eval_dtype']) for eps in epses]
    sr = model(lr=lr, z=None, epses=epses).float()
    return sr, [eps.float() for eps in epses]


@torch.no_grad()
def get_sr(model, opt, lq, is_rrdb=False, **kwargs):
    assert not model.training 
    if is_rrdb:
        return model(lq.to(opt['device'], opt['eval_dtype'])).float()
    else:
        return get_sr_with_epses(model, opt, lq, **kwargs)[0]


def compute_validation_metrics(model, opt, dataset, pad_factor=2, epses=None, is_rrdb=False):
    measure = Measure(use_gpu=True)
    df = None
    for (lr, hr) in dataset:
        lr, hr = np.array(lr), np.array(hr)
        h, w, _ = lr.shape
        lq_orig = lr.copy()
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                       right=int(np.ceil(w / pad_factor) * pad_factor - w))
        lr_t = t(lr)
        
        if epses is not None:
            tmp_epses = get_epses(opt, 1, lr_t.shape, opt['heat'])
            repeated_epses = []
            for i, tmp_eps in enumerate(tmp_epses):
                eps = epses[i].repeat(1, 1,16,16)[:, :, :tmp_eps.shape[2],:tmp_eps.shape[3]]
                repeated_epses.append(eps)
                
            sr_t = get_sr(model, opt, lq=lr_t, is_rrdb=False, epses=repeated_epses)
        else:
            sr_t = get_sr(model, opt, lq=lr_t, is_rrdb=is_rrdb)
        sr = rgb(sr_t)[:hr.shape[0], :hr.shape[1]]
        
        meas = OrderedDict()
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)
        lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
        meas['LRC PSNR'] = psnr(lq_orig, lr_reconstruct_rgb)
        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])
    return df.mean()


def compute_metrics_on_train_patches(model, opt, lrs, gts, epses=None, is_rrdb=False):
    measure = Measure(use_gpu=True)
    if epses is not None:
        epses = [eps.repeat(len(lrs), 1, 1, 1) for eps in epses]
        srs = get_sr(model, opt, lq=lrs, is_rrdb=False, epses=epses)
    else:
        srs = get_sr(model, opt, lq=lrs, is_rrdb=is_rrdb)

    df = None
    for (lr, sr, gt) in zip(lrs, srs, gts):
        lr, sr, hr = rgb(lr), rgb(sr), rgb(gt)
        meas = OrderedDict()
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

        lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
        meas['LRC PSNR'] = psnr(lr, lr_reconstruct_rgb)
        
        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])
    return df.mean()


def compute_metrics_on_patches(model, opt, lr_dir, hr_dir, epses=None, is_rrdb=False):
    measure = Measure(use_gpu=True)
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))
    
    df = None
    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
        hr = imread(hr_path)
        
        local_df = None
        lr_crops, hr_crops = patchify(hr, lr, opt['GT_size'], opt['scale'])
        for hr_crop, lr_crop in zip(hr_crops, lr_crops):
            lr_t = t(lr_crop)
            h_crop, w_crop, _ = lr_crop.shape

            sr_t = get_sr(model, opt, lq=lr_t, is_rrdb=is_rrdb, epses=epses)
            sr = rgb(sr_t)[:h_crop * opt['scale'], :w_crop * opt['scale']]

            meas = OrderedDict(name=idx_test)
            meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr_crop)

            lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
            meas['LRC PSNR'] = psnr(lr_crop, lr_reconstruct_rgb)
            local_df = pd.DataFrame([meas]) if local_df is None else pd.concat([pd.DataFrame([meas]), local_df])
            
        df = local_df if df is None else pd.concat([local_df, df])
        str_out = format_measurements(local_df.mean())
        print(str_out)

    str_out = format_measurements(df.mean())
    print('Mean: ' + str_out)


def compute_metrics(model, opt, lr_dir, hr_dir, epses=None, is_rrdb=False, 
                    conf='tmp', test_dir='../../results/tmp/'):
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=True)

    fname = f'measure_full.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    scale = opt['scale']

    pad_factor = 2

    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
        hr = imread(hr_path)

        # Pad image to be % 2
        h, w, c = lr.shape
        lq_orig = lr.copy()
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))
        lr_t = t(lr)

        if df is not None and len(df[(df['heat'] == opt['heat']) & (df['name'] == idx_test)]) == 1:
            continue

        if epses is not None:
            tmp_epses = get_epses(opt, 1, lr_t.shape, opt['heat'])
            repeated_epses = []
            for i, tmp_eps in enumerate(tmp_epses):
                eps = epses[i].repeat(1,1,16,16)[:, :, :tmp_eps.shape[2],:tmp_eps.shape[3]]
                repeated_epses.append(eps)
                
            sr_t = get_sr(model, opt, lq=lr_t, epses=repeated_epses)
        else:
            sr_t = get_sr(model, opt, lq=lr_t, is_rrdb=is_rrdb)
        sr = rgb(sr_t)[:h * scale, :w * scale]

        path_out_sr = os.path.join(test_dir, "{:0.2f}".format(opt['heat']).replace('.', ''), "{:06d}.png".format(idx_test))
        imwrite(path_out_sr, sr)

        meas = OrderedDict(conf=conf, heat=opt['heat'], name=idx_test)
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

        lr_reconstruct_rgb = imresize(sr, 1 / opt['scale'])
        meas['LRC PSNR'] = psnr(lq_orig, lr_reconstruct_rgb)

        str_out = format_measurements(meas)
        print(str_out)

        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])
        df.to_csv(path_out_measures + "_", index=False)
        os.rename(path_out_measures + "_", path_out_measures)

    df.to_csv(path_out_measures, index=False)
    os.rename(path_out_measures, path_out_measures_final)

    str_out = format_measurements(df.mean())
    print(f"Results in: {path_out_measures_final}")
    print('Mean: ' + str_out)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.3f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


def eval_diversity(model, opt, lr_dir, num_samples=8):
    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    measure = Measure(use_gpu=True)
    pad_factor = 2

    diversity = 0.0
    for lr_path in tqdm(lr_paths):
        lr = imread(lr_path)

        # Pad image to be % 2
        h, w, _ = lr.shape
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))
        lr_t = t(lr)

        srs = []
        for _ in range(num_samples):
            sr = rgb(get_sr(model, opt, lq=lr_t))
            srs.append(sr[:h * opt['scale'], :w * opt['scale']])

        lpips = 0.0
        for i in range(num_samples):
            for j in range(num_samples):
                if i == j: continue 
                lpips += measure.measure(srs[j], srs[i])[2]
        
        diversity += lpips / (num_samples * (num_samples - 1))
    return diversity / len(lr_paths)


@torch.no_grad()
def get_inference_time(model, opt, demo_sample, repetitions=100, is_rrdb=False):
    assert not model.training
    
    # Prepare input 
    lq = demo_sample.to(opt['device'], opt['eval_dtype'])
    epses = get_epses(opt, lq.shape[0], lq.shape, opt['heat'])
    epses = [eps.to(opt['device'], opt['eval_dtype']) for eps in epses]
    torch.cuda.empty_cache()
    
    # Create timers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
        if is_rrdb:
            model(lq) 
        else:
            model(lr=lq, z=None, epses=epses)

    # MEASURE PERFORMANCE
    for rep in tqdm(range(repetitions)):
        starter.record()
        if is_rrdb:
            model(lq) 
        else:
            model(lr=lq, z=None, epses=epses)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn
