import os
import os.path as osp
import random

import cv2
import numpy as np
import mmcv
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette

from . import transforms as tf
from .utils import colormap


def crop(img_temp, dim, new_p=True, h_p=0, w_p=0):
    h = img_temp.shape[0]
    w = img_temp.shape[1]
    trig_h = trig_w = False
    if (h > dim):
        if (new_p):
            h_p = int(random.uniform(0, 1) * (h - dim))
        if len(img_temp.shape) == 2:
            img_temp = img_temp[h_p:h_p + dim, :]
        else:
            img_temp = img_temp[h_p:h_p + dim, :, :]
    elif (h < dim):
        trig_h = True
    if (w > dim):
        if (new_p):
            w_p = int(random.uniform(0, 1) * (w - dim))
        if len(img_temp.shape) == 2:
            img_temp = img_temp[:, w_p:w_p + dim]
        else:
            img_temp = img_temp[:, w_p:w_p + dim, :]
    elif (w < dim):
        trig_w = True
    if (trig_h or trig_w):
        if len(img_temp.shape) == 2:
            pad = np.zeros((dim, dim))
            pad[:img_temp.shape[0], :img_temp.shape[1]] = img_temp
        else:
            pad = np.zeros((dim, dim, 3))
            pad[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        return (pad, h_p, w_p)
    else:
        return (img_temp, h_p, w_p)


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST)


def flip(img, flip_p):
    if flip_p > 0.5:
        return np.fliplr(img)
    else:
        return img


def transform(img, mask, pseudo_mask, crop_size, scale_range):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    flip_p = np.random.uniform(0, 1)
    img_temp = scale_im(img, scale)
    raw_img_temp = scale_im(img, scale)
    gt_temp = scale_gt(mask, scale)
    pseudo_gt_temp = scale_gt(pseudo_mask, scale)

    img_temp = flip(img_temp, flip_p)
    gt_temp = flip(gt_temp, flip_p)
    pseudo_gt_temp = flip(pseudo_gt_temp, flip_p)

    img_temp = img_temp.astype(np.float32)

    img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
    img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
    img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

    img_temp, img_temp_h_p, img_temp_w_p = crop(img_temp, crop_size)
    gt_temp = crop(gt_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]
    pseudo_gt_temp = crop(pseudo_gt_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]
    raw_img = crop(raw_img_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]

    return img_temp, gt_temp, pseudo_gt_temp, raw_img


def convert(img, alpha=1, beta=0):
    """Multiple with alpha and add beat with clip."""
    img = img.astype(np.float32) * alpha + beta
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def brightness(img, brightness_delta):
    """Brightness distortion."""
    if random.randint(0, 2):
        return convert(img, beta=random.uniform(-brightness_delta, brightness_delta))
    return img


def contrast(img, contrast_lower, contrast_upper):
    """Contrast distortion."""
    if random.randint(0, 2):
        return convert(img, alpha=random.uniform(contrast_lower, contrast_upper))
    return img


def saturation(img, saturation_lower, saturation_upper):
    """Saturation distortion."""
    if random.randint(0, 2):
        img = mmcv.bgr2hsv(img)
        img[:, :, 1] = convert(img[:, :, 1], alpha=random.uniform(saturation_lower, saturation_upper))
        img = mmcv.hsv2bgr(img)
    return img


def hue(img, hue_delta):
    """Hue distortion."""
    if random.randint(0, 2):
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)) % 180
        img = mmcv.hsv2bgr(img)
    return img


def colorjitter(img, p=0.8, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
    if random.random() < p:
        contrast_lower, contrast_upper = contrast_range
        saturation_lower, saturation_upper = saturation_range

        # random brightness
        img = brightness(img, brightness_delta)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = contrast(img, contrast_lower, contrast_upper)

        # random saturation
        img = saturation(img, saturation_lower, saturation_upper)

        # random hue
        img = hue(img, hue_delta)

        # random contrast
        if mode == 0:
            img = contrast(img, contrast_lower, contrast_upper)

    return img


def blur(img, p=0.5):
    if random.random() < p:
        img = cv2.blur(img, (5, 5))

    return img


def transform_strongaug(img, mask, pseudo_mask, crop_size, scale_range):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    flip_p = np.random.uniform(0, 1)
    img_temp = scale_im(img, scale)
    raw_img_temp = scale_im(img, scale)
    gt_temp = scale_gt(mask, scale)
    pseudo_gt_temp = scale_gt(pseudo_mask, scale)

    img_temp = flip(img_temp, flip_p)
    gt_temp = flip(gt_temp, flip_p)
    pseudo_gt_temp = flip(pseudo_gt_temp, flip_p)

    img_s_temp = colorjitter(img_temp.copy())

    img_temp = img_temp.astype(np.float32)
    img_s_temp = img_s_temp.astype(np.float32)

    img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
    img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
    img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

    img_s_temp[:, :, 0] = (img_s_temp[:, :, 0] / 255. - 0.485) / 0.229
    img_s_temp[:, :, 1] = (img_s_temp[:, :, 1] / 255. - 0.456) / 0.224
    img_s_temp[:, :, 2] = (img_s_temp[:, :, 2] / 255. - 0.406) / 0.225

    img_temp, img_temp_h_p, img_temp_w_p = crop(img_temp, crop_size)
    img_s_temp = crop(img_s_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]
    gt_temp = crop(gt_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]
    pseudo_gt_temp = crop(pseudo_gt_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]
    raw_img_temp = crop(raw_img_temp, crop_size, False, img_temp_h_p, img_temp_w_p)[0]

    return img_temp, img_s_temp, gt_temp, pseudo_gt_temp, raw_img_temp


class VOC(Dataset):

    CLASSES = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted-plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ]

    CLASS_IDX = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'potted-plant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tv/monitor': 20,
        'unlabeled': 255
    }

    CLASS_IDX_INV = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted-plant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tv/monitor',
        255: 'unlabeled'
    }

    NUM_CLASSES = 21

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette


class VOCPseudoGT(VOC):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('./data/pascal_voc/')):
        super(VOCPseudoGT, self).__init__()

        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.pseudo_gt_path = cfg.DATASET.PSEUDO_GT_PATH
        self.crop_size = cfg.DATASET.CROP_SIZE
        self.scale_range = (cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_aug_id.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_id.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val_id.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.image_folder = osp.join(self.root, 'VOC2012/JPEGImages')
        self.mask_folder = osp.join(self.root, 'VOC2012/SegmentationClassAug') if self.split not in ['test'] else None
        self.pseudo_mask_folder = self.pseudo_gt_path if self.split not in ['test'] else None

        self.images = []
        self.masks = []
        self.pseudo_masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                name = line.strip()
                _image = osp.join(self.image_folder, f'{name}.jpg')
                self.images.append(_image)
                if self.mask_folder is not None:
                    _mask = osp.join(self.mask_folder, f'{name}.png')
                    self.masks.append(_mask)
                if self.pseudo_mask_folder is not None:
                    _pseudo_mask = osp.join(self.pseudo_mask_folder, f'{name}.png')
                    self.pseudo_masks.append(_pseudo_mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dataset_dict = {}
        dataset_dict.update({'img': img, 'filename': osp.basename(self.images[index]), 'seg_fileds': []})

        if self.mask_folder is not None:
            mask = np.asarray(Image.open(self.masks[index]))

            unique_labels = np.unique(mask)

            # ambigious
            unique_labels = unique_labels[unique_labels != 255]
            unique_labels = unique_labels[unique_labels != 0]
            unique_labels -= 1  # shifting since no BG class
            # ignoring BG
            labels = torch.zeros(self.NUM_CLASSES - 1)

            labels[unique_labels.tolist()] = 1

            dataset_dict['pix_gt'] = mask
            dataset_dict['img_gt'] = labels
            dataset_dict['seg_fileds'].append('pix_gt')

        if self.pseudo_mask_folder is not None:
            pseudo_mask = np.asarray(Image.open(self.pseudo_masks[index]))
            dataset_dict['pseudo_pix_gt'] = pseudo_mask
            dataset_dict['seg_fileds'].append('pseudo_pix_gt')

        if not self.test_mode:
            img, pix_gt, pseudo_pix_gt, raw_img = transform(img, mask, pseudo_mask, self.crop_size, self.scale_range)

            dataset_dict['img'] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
            dataset_dict['pix_gt'] = torch.from_numpy(np.ascontiguousarray(pix_gt)).float()
            dataset_dict['pseudo_pix_gt'] = torch.from_numpy(np.ascontiguousarray(pseudo_pix_gt)).float()
            dataset_dict['raw_img'] = torch.from_numpy(np.ascontiguousarray(raw_img)).permute(2, 0, 1).float()
        else:
            # general resize, normalize and toTensor
            dataset_dict = tf.MaskNormalize(self.MEAN, self.STD)(dataset_dict)

        return dataset_dict


class VOCPseudoGTDual(VOC):

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('./data/pascal_voc/')):
        super(VOCPseudoGTDual, self).__init__()

        self.root = root
        self.split = split
        self.test_mode = test_mode
        self.pseudo_gt_path = cfg.DATASET.PSEUDO_GT_PATH
        self.crop_size = cfg.DATASET.CROP_SIZE
        self.scale_range = (cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_aug_id.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_id.txt')
        elif self.split == 'val':
            # _split_f = os.path.join(self.root, 'train_id.txt')
            _split_f = os.path.join(self.root, 'val_id.txt')
        elif self.split == 'test':
            _split_f = os.path.join(self.root, 'test_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.image_folder = osp.join(self.root, 'VOC2012/JPEGImages')
        self.mask_folder = osp.join(self.root, 'VOC2012/SegmentationClassAug')
        self.pseudo_mask_folder = self.pseudo_gt_path if self.split not in ['test'] else None

        self.images = []
        self.masks = []
        self.pseudo_masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                name = line.strip()
                _image = osp.join(self.image_folder, f'{name}.jpg')
                _mask = osp.join(self.mask_folder, f'{name}.png')
                self.images.append(_image)
                self.masks.append(_mask)
                if self.pseudo_mask_folder is not None:
                    _pseudo_mask = osp.join(self.pseudo_mask_folder, f'{name}.png')
                    self.pseudo_masks.append(_pseudo_mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(self.masks[index]))

        unique_labels = np.unique(mask)

        # ambigious
        unique_labels = unique_labels[unique_labels != 255]
        unique_labels = unique_labels[unique_labels != 0]
        unique_labels -= 1  # shifting since no BG class
        # ignoring BG
        labels = torch.zeros(self.NUM_CLASSES - 1)

        labels[unique_labels.tolist()] = 1

        dataset_dict = {}
        dataset_dict.update({
            'img': img,
            'pix_gt': mask,
            'img_gt': labels,
            'filename': osp.basename(self.images[index]),
            'seg_fileds': ['pix_gt']
        })

        if self.pseudo_mask_folder is not None:
            pseudo_mask = np.asarray(Image.open(self.pseudo_masks[index]))
            dataset_dict['pseudo_pix_gt'] = pseudo_mask
            dataset_dict['seg_fileds'].append('pseudo_pix_gt')

        if not self.test_mode:
            img, img_s, pix_gt, pseudo_pix_gt, raw_img = transform_strongaug(img, mask, pseudo_mask, self.crop_size,
                                                                             self.scale_range)
            dataset_dict['img'] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
            dataset_dict['img_s'] = torch.from_numpy(np.ascontiguousarray(img_s)).permute(2, 0, 1).float()
            dataset_dict['pix_gt'] = torch.from_numpy(np.ascontiguousarray(pix_gt)).float()
            dataset_dict['pseudo_pix_gt'] = torch.from_numpy(np.ascontiguousarray(pseudo_pix_gt)).float()
            dataset_dict['raw_img'] = torch.from_numpy(np.ascontiguousarray(raw_img)).permute(2, 0, 1).float()
        else:
            # general resize, normalize and toTensor
            dataset_dict = tf.MaskNormalize(self.MEAN, self.STD)(dataset_dict)

        return dataset_dict
