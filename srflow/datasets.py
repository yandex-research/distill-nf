import os
import glob
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

from utils.imresize import imresize
from utils.util import imread, impad, paired_random_crop


class DF2K(Dataset):
    """DF2K dataset --- a merged training dataset of DIV2K and Flickr2K
    Parameters:
        root (str)  -- path to the DIV2K and Flickr2K datasets
        split (str) -- dataset split: 'train' | 'val' | 'test'
        scale (int) -- SR scaling factor
        GT_size (int) -- HR patch size
        totensor (bool) -- whether to transform images to tensors or not 
    """

    def __init__(self, root='./data', split='train', scale=4, GT_size=160, totensor=False):
        assert split in ['train', 'val', 'test'], f"Wrong data split: {split}"
        img_path = os.path.join(root, 'DIV2K')
        self.split = split
        self.scale = scale
        self.GT_size = GT_size

        if split == 'train':
            self.hr_img_path = os.path.join(img_path, f'DIV2K_{split}_HR_sub')
            self.hr_image_list = sorted(glob.glob(os.path.join(self.hr_img_path, '*')))
            self.hr_img_path = os.path.join(root, 'Flickr2K/Flickr2K_HR_sub')
            self.hr_image_list.extend(sorted(glob.glob(os.path.join(self.hr_img_path, '*'))))
        elif split == 'val':
            self.hr_img_path = os.path.join(root, 'Flickr2K/Flickr2K_HR_val')
            self.hr_image_list = sorted(glob.glob(os.path.join(self.hr_img_path, '*')))
            self.lr_img_path = os.path.join(root, f'Flickr2K/Flickr2K_LR_bicubic/X{scale}_val')
            self.lr_image_list = sorted(glob.glob(os.path.join(self.lr_img_path, '*')))
        elif split == 'test':
            self.hr_img_path = os.path.join(img_path, f'div2k-validation-modcrop8-gt')
            self.hr_image_list = sorted(glob.glob(os.path.join(self.hr_img_path, '*')))
            self.lr_img_path = os.path.join(img_path, f'div2k-validation-modcrop8-x{scale}')
            self.lr_image_list = sorted(glob.glob(os.path.join(self.lr_img_path, '*')))
       
        self.transform = transforms.ToTensor() if totensor else None

    def __getitem__(self, index):
        HR_image = imread(self.hr_image_list[index])
        
        if self.split == 'train':
            LR_image = imresize(HR_image, scalar_scale=1. / self.scale)
            HR_crop, LR_crop = paired_random_crop(HR_image, LR_image, self.GT_size, scale=self.scale)
            HR_crop, LR_crop = random_flip(HR_crop, LR_crop)
            HR_image, LR_image = HR_crop, LR_crop 
        elif self.split == 'val':
            LR_image = imresize(HR_image, scalar_scale=1. / self.scale)
        elif self.split == 'test':
            LR_image = imread(self.lr_image_list[index])
            h, w, c = LR_image.shape
            self.pad_factor = 2
            LR_image = impad(LR_image, bottom=int(np.ceil(h / self.pad_factor) * self.pad_factor - h),
                                       right=int(np.ceil(w / self.pad_factor) * self.pad_factor - w))
                                       
        if self.transform:
            LR_image = self.transform(LR_image)
            HR_image = self.transform(HR_image)
        else:
            LR_image = Image.fromarray(LR_image)
            HR_image = Image.fromarray(HR_image)
        return LR_image, HR_image

    def __len__(self):
        return len(self.hr_image_list)


class EvalDataset(Dataset):
    """ Images for qualitative evaluation. You provide HR images and obtain (LR, HR) pairs 
    Parameters:
        image_path (str) -- path to your HR images
        scale (int)      -- SR scaling factor
        totensor (bool)  -- whether to transform images to tensors or not 
    """
    def __init__(self, image_path='./data/BSD100', scale=4, totensor=False):
        self.scale = scale
        self.pad_factor = 2

        self.hr_image_list = sorted(glob.glob(os.path.join(image_path, '*')))
        self.transform = transforms.ToTensor() if totensor else None

    def __getitem__(self, index):
        HR_image = Image.open(self.hr_image_list[index]).convert('RGB')
        HR_image = np.asarray(HR_image)
        LR_image = imresize(HR_image, scalar_scale=1. / self.scale)
        h, w, c = LR_image.shape
        LR_image = impad(LR_image, bottom=int(np.ceil(h / self.pad_factor) * self.pad_factor - h),
                                    right=int(np.ceil(w / self.pad_factor) * self.pad_factor - w))

        if self.transform is None:
            return Image.fromarray(LR_image), Image.fromarray(HR_image)

        LR_image = self.transform(np.array(LR_image))
        HR_image = self.transform(np.array(HR_image))
        return LR_image, HR_image


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]
