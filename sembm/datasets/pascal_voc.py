import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette

from . import transforms as tf
from .seg_randaugment import SegRandomAugment
from .utils import colormap


class PascalVOC(Dataset):

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


class VOCSegmentation(PascalVOC):

    def __init__(self, cfg, split, root=os.path.expanduser('./data/pascal_voc/')):
        super(VOCSegmentation, self).__init__()

        self.root = root
        self.split = split

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_aug_id.txt')
        elif self.split == 'train_voc':
            _split_f = os.path.join(self.root, 'train_id.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.images = []
        self.masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                name = line.strip()
                _image = osp.join(self.root, 'VOC2012/JPEGImages', f'{name}.jpg')
                _mask = osp.join(self.root, 'VOC2012/SegmentationClassAug', f'{name}.png')
                self.images.append(_image)
                self.masks.append(_mask)

        if self.split in ['train', 'train_voc']:
            self.transform = tf.Compose([
                tf.MaskRandResizedCrop(
                    size=cfg.DATASET.CROP_SIZE, scale=(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)),
                tf.MaskHFlip(),
                tf.MaskColourJitter(p=0.5, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                tf.MaskRandGrayscale(p=0.2),
                tf.MaskNormalize(self.MEAN, self.STD)
            ])
            # NOTE: This type of transform seems to have higher performance
            # self.transform = tf.Compose(
            #     [tf.MaskFixResize(cfg.DATASET.CROP_SIZE),
            #      tf.MaskNormalize(self.MEAN, self.STD)])
        elif self.split in ['val', 'test']:
            self.transform = tf.Compose([tf.MaskNormalize(self.MEAN, self.STD)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX['unlabeled']:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        labels = torch.zeros(self.NUM_CLASSES - 1)
        if unique_labels[0] == self.CLASS_IDX['background']:
            unique_labels = unique_labels[1:]
        unique_labels -= 1  # shifting since no BG class

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        labels[unique_labels.tolist()] = 1

        dataset_dict = {}
        dataset_dict.update({
            'img': image,
            'pix_gt': mask,
            'img_gt': labels,
            'filename': osp.basename(self.images[index]),
        })

        # general resize, normalize and toTensor
        dataset_dict = self.transform(dataset_dict)

        return dataset_dict


class VOCSegmentation_Augmentation(VOCSegmentation):

    def __init__(self, cfg, split, root, reduce_zero_label=True):
        super(VOCSegmentation_Augmentation, self).__init__(split, root)
        self.reduce_zero_label = reduce_zero_label

        self.weak_transform = tf.Compose([
            tf.MaskRandResizedCrop(size=cfg.DATASET.CROP_SIZE, scale=(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)),
            tf.MaskHFlip(),
            tf.MaskColourJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            tf.MaskRandGrayscale(p=0.2),
            tf.MaskNormalize(self.MEAN, self.STD)
        ])
        self.strong_transform = tf.Compose([
            tf.MaskRandResizedCrop(size=cfg.DATASET.CROP_SIZE, scale=(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)),
            tf.MaskHFlip(),
            SegRandomAugment(3, 5),
            tf.MaskNormalize(self.MEAN, self.STD)
        ])

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX['ambiguous']:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        if unique_labels[0] == self.CLASS_IDX['background']:
            unique_labels = unique_labels[1:]
        unique_labels -= 1  # shifting since no BG class

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        labels = torch.zeros(self.NUM_CLASS - 1)
        labels[unique_labels.tolist()] = 1

        dataset_dict = {
            'height': image.height,
            'width': image.width,
            'image_gt': labels,
            'index': index,
            'img_filename': self.images[index],
            'gt_filename': self.masks[index],
        }

        weak_dict = {
            'image': image,
            'pixel_gt': mask,
        }

        strong_dict = {
            'image': image,
            'pixel_gt': mask,
        }

        weak_dict = self.weak_transform(weak_dict)
        strong_dict = self.strong_transform(strong_dict)

        dataset_dict.update({
            'image_weakaug': weak_dict['image'],
            'pixel_gt_weakaug': weak_dict['pixel_gt'],
            'crop_box_weakaug': torch.tensor(weak_dict['crop_box']),
            'hflip_weakaug': weak_dict['hflip'],
            'image_strongaug': strong_dict['image'],
            'pixel_gt_strongaug': strong_dict['pixel_gt'],
            'crop_box_strongaug': torch.tensor(strong_dict['crop_box']),
            'hflip_strongaug': strong_dict['hflip'],
        })

        return dataset_dict


def load_pascal_voc(cfg):
    train_dataset = VOCSegmentation(cfg, split='train', root='./data')
    test_dataset = VOCSegmentation(cfg, split='val', root='./data')

    return train_dataset, test_dataset


def load_pascal_voc_dual(cfg):
    train_dataset = VOCSegmentation_Augmentation(cfg, split='train', root='./data')
    test_dataset = VOCSegmentation(cfg, split='val', root='./data')

    return train_dataset, test_dataset
