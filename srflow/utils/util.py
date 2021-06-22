import glob
import os
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict

import natsort
import numpy as np
import cv2
import torch
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


######################
# Load SRFlow teacher
######################

# Create SRFlow teacher adapted for efficient inference
def load_teacher(conf_path, dtype=torch.float32, device='cuda'):
    import utils.options as option
    from models import create_model
    opt = option.parse(conf_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    teacher = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    teacher.load_network(load_path=model_path, network=teacher.netG)
    teacher_model = teacher.netG.module.to(device, dtype).train(False)

    opt['device'] = device
    opt['flowUpsamplerNet'] = {}
    opt['flowUpsamplerNet']['C'] = teacher_model.flowUpsamplerNet.C
    opt['flowUpsamplerNet']['scaleH'] = teacher_model.flowUpsamplerNet.scaleH
    opt['flowUpsamplerNet']['scaleW'] = teacher_model.flowUpsamplerNet.scaleW
    return teacher_model, opt


####################
# miscellaneous
####################

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def num_params(model):
    return sum((param.numel() for param in model.parameters()))


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image utils
####################


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     if hasattr(tensor, 'detach'):
#         tensor = tensor.detach()
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type)


####################
# metrics
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def get_resume_paths(opt):
    resume_state_path = None
    resume_model_path = None
    if opt.get('path', {}).get('resume_state', None) == "auto":
        wildcard = os.path.join(opt['path']['training_state'], "*")
        paths = natsort.natsorted(glob.glob(wildcard))
        if len(paths) > 0:
            resume_state_path = paths[-1]
            resume_model_path = resume_state_path.replace('training_state', 'models').replace('.state', '_G.pth')
    else:
        resume_state_path = opt.get('path', {}).get('resume_state')
    return resume_state_path, resume_model_path


def opt_get(opt, keys, default=None):
    if opt is None:
        return default
    ret = opt
    for k in keys:
        ret = ret.get(k, None)
        if ret is None:
            return default
    return ret


##############
# Data utils #
##############

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    """Paired random crop.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}).')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def patchify(img_gt, img_lq, gt_patch_size, scale):
    """ It patchifies lq and gt images on non-overlapping patches.
    Args:
        img_gt (ndarray): GT image.
        img_lq (ndarray): LQ image.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
    Returns:
        list[ndarray]: all consequential patches for GT and LQ images.
    """
    h_lq, w_lq, _ = img_lq.shape
    h_gt, w_gt, _ = img_gt.shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}).')

    img_lq_patches = []
    img_gt_patches = []
    for top in range(0, h_lq - lq_patch_size, lq_patch_size):
        for left in range(0, w_lq - lq_patch_size, lq_patch_size):
            lq_patch = img_lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            img_lq_patches.append(lq_patch)
            # crop corresponding gt patch
            top_gt, left_gt = int(top * scale), int(left * scale)
            gt_patch = img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
            img_gt_patches.append(gt_patch)
    return img_lq_patches, img_gt_patches