import torch
import torch.nn.functional as F


def histc(array, bins, min, max):
    try:
        return torch.histc(array, bins, min, max)
    except:
        count = torch.zeros(bins)

        for i in range(min, max + 1):
            count[i] = torch.sum(array == i)

        return count


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x
