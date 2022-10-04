import random
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as tf
import torchvision.transforms.functional as F


class Compose:

    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, results):
        for t in self.segtransform:
            results = t(results)

        return results


class MaskRandResizedCrop:

    def __init__(self, size, scale):
        self.rnd_crop = tf.RandomResizedCrop(size, scale=scale)

    def get_params(self, image):
        return self.rnd_crop.get_params(image, self.rnd_crop.scale, self.rnd_crop.ratio)

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        i, j, h, w = self.get_params(image)

        image = F.resized_crop(image, i, j, h, w, self.rnd_crop.size, Image.CUBIC)
        pixel_gt = F.resized_crop(pixel_gt, i, j, h, w, self.rnd_crop.size, Image.NEAREST)

        results['img'] = image
        results['pix_gt'] = pixel_gt
        results['crop_box'] = [j, i, w, h]  # xywh

        return results


class MaskFixResize:

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        image = F.resize(image, self.size)
        pixel_gt = F.resize(pixel_gt, self.size, Image.NEAREST)

        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskHFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        results['hflip'] = False
        if random.random() < self.p:
            image = F.hflip(image)
            pixel_gt = F.hflip(pixel_gt)
            results['hflip'] = True

        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskNormalize:

    def __init__(self, mean, std):
        self.norm = tf.Normalize(mean, std)

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        raw_image = image.copy()
        raw_image = F.to_tensor(raw_image)
        image = F.to_tensor(image)
        image = self.norm(image)
        pixel_gt = self.__toByteTensor(pixel_gt)

        results['raw_img'] = raw_image * 255
        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskCustomNormalize:

    def __init__(self, mean, std):
        self.norm = tf.Normalize(mean, std)

    def __toByteTensor(self, pic):
        return torch.from_numpy(np.array(pic, np.int32, copy=False))

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        raw_image = image.copy()
        raw_image = F.to_tensor(raw_image)
        image = F.to_tensor(image) * 255
        image = image[(2, 1, 0), :, :]
        image[0, :, :] = image[0, :, :] - 104.008
        image[1, :, :] = image[1, :, :] - 116.669
        image[2, :, :] = image[2, :, :] - 122.675
        pixel_gt = self.__toByteTensor(pixel_gt)

        results['raw_img'] = raw_image * 255
        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskToTensor:

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        gt_labels = torch.arange(0, 21)
        gt_labels = gt_labels.unsqueeze(-1).unsqueeze(-1)
        pixel_gt = pixel_gt.unsqueeze(0).type_as(gt_labels)
        pixel_gt = torch.eq(pixel_gt, gt_labels).float()

        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskColourJitter:

    def __init__(self, p=0.5, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.p = p
        self.jitter = tf.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        if random.random() < self.p:
            image = self.jitter(image)

        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


def _get_image_num_channels(img) -> int:
    if isinstance(img, Image.Image):
        return 1 if img.mode == 'L' else 3
    raise TypeError("Unexpected type {}".format(type(img)))


def to_grayscale(img, num_output_channels):
    if not isinstance(img, Image.Image):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


class MaskRandGrayscale:

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, results):
        image = results['img']
        pixel_gt = results['pix_gt']

        num_output_channels = _get_image_num_channels(image)
        if torch.rand(1) < self.p:
            image, pixel_gt = to_grayscale(image, num_output_channels=num_output_channels), pixel_gt

        results['img'] = image
        results['pix_gt'] = pixel_gt

        return results


class MaskPad:

    def __init__(self, pad_size):
        if isinstance(pad_size, int):
            pad_size = (pad_size, pad_size)
        self.pad_size = pad_size

    def __call__(self, results):
        image = results['img']
        pix_gt = results['pix_gt']

        h, w = image.height, image.width
        pad_h, pad_w = max(self.pad_size[0] - h, 0), max(self.pad_size[1] - w, 0)
        image = F.pad(image, padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')
        pix_gt = F.pad(pix_gt, padding=(0, 0, pad_w, pad_h), fill=255, padding_mode='constant')

        results['img'] = image
        results['pix_gt'] = pix_gt

        return results
