import random

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np


def AutoContrast(img, mask, _):
    return ImageOps.autocontrast(img), mask


def Brightness(img, mask, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v), mask


def Color(img, mask, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v), mask


def Contrast(img, mask, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v), mask


def Equalize(img, mask, _):
    return ImageOps.equalize(img), mask


def Invert(img, mask, _):
    return ImageOps.invert(img), mask


def Identity(img, mask, v):
    return img, mask


def Posterize(img, mask, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v), mask


def Rotate(img, mask, v):  # [-30, 30]
    # assert -30 <= v <= 30
    # if random.random() > 0.5:
    #     v = -v
    return img.rotate(v), mask.rotate(v)


def Sharpness(img, mask, v):  # [0.1,1.9]
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v), mask


def ShearX(img, mask, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #     v = -v
    return img.transform(img.size, Image.AFFINE,
                         (1, v, 0, 0, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, mask, v):  # [-0.3, 0.3]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #     v = -v
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, 0, v, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #     v = -v
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, v, 0, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert v >= 0.0
    # if random.random() > 0.5:
    #     v = -v
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, v, 0, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert -0.3 <= v <= 0.3
    # if random.random() > 0.5:
    #     v = -v
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, 0, 0, 1, v)), mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    # assert 0 <= v
    # if random.random() > 0.5:
    #     v = -v
    return img.transform(img.size, Image.AFFINE,
                         (1, 0, 0, 0, 1, v)), mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, mask, v):  # [0, 256]
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Cutout(img, mask, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img, mask

    v = v * img.size[0]
    return CutoutAbs(img, mask, v)


def CutoutAbs(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    mask = mask.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    ImageDraw.Draw(mask).rectangle(xy, 0)
    return img, mask


def augment_list():
    al = [(AutoContrast, 0, 1), (Brightness, 0.05, 0.95), (Color, 0.05, 0.95), (Contrast, 0.05, 0.95), (Equalize, 0, 1),
          (Identity, 0, 1), (Posterize, 4, 8), (Rotate, -30, 30), (Sharpness, 0.05, 0.95), (ShearX, -0.3, 0.3),
          (ShearY, -0.3, 0.3), (Solarize, 0, 256), (TranslateX, -0.3, 0.3), (TranslateY, -0.3, 0.3)]
    return al


class SegRandomAugment:

    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, results):
        img = results['img']
        mask = results['pix_gt']

        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img, mask = op(img, mask, val)
        cutout_val = random.random() * 0.5
        img, mask = Cutout(img, mask, cutout_val)  # for fixmatch

        results['img'] = img
        results['pix_gt'] = mask

        return results
