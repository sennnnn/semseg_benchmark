import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImagePalette

from . import transforms as tf
from .seg_randaugment import SegRandomAugment
from .utils import colormap


class COCO(Dataset):

    CLASSES = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    NUM_CLASSES = 81

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


class COCOSegmentation(COCO):

    def __init__(self, cfg, split, root=os.path.expanduser('./data/coco/')):
        super(COCOSegmentation, self).__init__()

        self.root = root
        self.split = split
        self.pseudo_gt_path = cfg.DATASET.PSEUDO_GT_PATH

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_aug_id.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.image_folder = osp.join(self.root, f'images/{self.split}2014')
        self.mask_folder = osp.join(self.root, f'SegmentationClass/{self.split}2014')
        self.pseudo_mask_folder = self.pseudo_gt_path

        self.images = []
        self.masks = []
        self.pseudo_masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                name = line.strip()
                _image = osp.join(self.image_folder, f'{name}.jpg')
                _mask = osp.join(self.mask_folder, f'{name}.png')
                _pseudo_mask = osp.join(self.pseudo_mask_folder, f'{name}.png')
                self.images.append(_image)
                self.masks.append(_mask)
                self.pseudo_masks.append(_pseudo_mask)

        if self.split in ['train', 'train_voc']:
            self.transform = tf.Compose([
                tf.MaskRandResizedCrop(
                    size=cfg.DATASET.CROP_SIZE, scale=(cfg.DATASET.SCALE_FROM, cfg.DATASET.SCALE_TO)),
                tf.MaskHFlip(),
                # tf.MaskColourJitter(p=0.5, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # tf.MaskRandGrayscale(p=0.2),
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
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        pseudo_mask = Image.open(self.pseudo_masks[index])

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == 255:
            unique_labels = unique_labels[:-1]

        # ignoring BG
        labels = torch.zeros(self.NUM_CLASSES - 1)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        unique_labels -= 1  # shifting since no BG class

        assert unique_labels.size > 0, 'No labels found in %s' % self.masks[index]
        labels[unique_labels.tolist()] = 1

        dataset_dict = {}
        dataset_dict.update({
            'img': img,
            'pix_gt': mask,
            'pseudo_pix_gt': pseudo_mask,
            'img_gt': labels,
            'filename': osp.basename(self.images[index]),
            'seg_fileds': ['pix_gt', 'pseudo_pix_gt']
        })

        # general resize, normalize and toTensor
        dataset_dict = self.transform(dataset_dict)

        return dataset_dict
