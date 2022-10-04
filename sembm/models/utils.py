import torch
import torch.nn.functional as F


def resize(x, size, mode='bilinear', align_corners=True):
    if mode == 'nearest':
        return F.interpolate(x, size, mode=mode)
    return F.interpolate(x, size, mode=mode, align_corners=align_corners)


def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x


def focal_loss(x, p=1, c=0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)


def pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= 0.7
    mask_max[:, 1:] *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
    pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, c, h, w)


def new_pseudo_gtmask(mask, cutoff_top=0.6, cutoff_low=0.2, eps=1e-8):
    """Convert continuous mask into binary mask"""
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max[:, :1] *= 0.7
    mask_max[:, 1:] *= cutoff_top
    # mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)

    # remove ambiguous pixels
    ambiguous = (pseudo_gt.sum(1) > 1.) + (pseudo_gt.sum(1) < 1.)
    pseudo_gt = torch.argmax(pseudo_gt, dim=1)
    pseudo_gt[ambiguous > 0] = 255
    # pseudo_gt = (1 - ambiguous) * pseudo_gt

    return pseudo_gt.view(bs, h, w)


def mask_loss_focal(preds, labels, alpha=0.25, gamma=2.0, ignore_idx=255):
    _alpha = alpha
    _gamma = gamma

    bs, ch, h, w = preds.shape

    alpha = torch.zeros((ch, ), device=preds.device)
    alpha[0] = _alpha
    alpha[1:] = (1 - _alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
    gamma = torch.tensor(_gamma, device=preds.device)

    preds = preds.permute(0, 2, 3, 1).contiguous()
    preds = preds[labels != ignore_idx]
    labels = labels[labels != ignore_idx]

    preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
    preds_softmax = torch.exp(preds_logsoft)  # softmax
    preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
    alpha = alpha.gather(0, labels.view(-1))

    loss = -torch.mul(torch.pow(
        (1 - preds_softmax), gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

    loss = torch.mul(alpha, loss.t())
    loss = loss.mean()

    return loss


def balanced_mask_loss_ce(mask, pseudo_gt, gt_labels, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

    # indices of the max classes
    mask_gt = torch.argmax(pseudo_gt, 1)

    # for each pixel there should be at least one 1
    # otherwise, ignore
    ignore_mask = pseudo_gt.sum(1) < 1.
    mask_gt[ignore_mask] = ignore_index

    # class weight balances the loss w.r.t. number of pixels
    # because we are equally interested in all classes
    bs, c, h, w = pseudo_gt.size()
    num_pixels_per_class = pseudo_gt.view(bs, c, -1).sum(-1)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
    class_weight = (pseudo_gt * class_weight[:, :, None, None]).sum(1).view(bs, -1)

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")
    loss = loss.view(bs, -1)

    # we will have the loss only for batch indices
    # which have all classes in pseudo mask
    gt_num_labels = gt_labels.sum(-1).type_as(loss) + 1  # + BG
    ps_num_labels = (num_pixels_per_class > 0).type_as(loss).sum(-1)
    batch_weight = (gt_num_labels == ps_num_labels).type_as(loss)

    loss = batch_weight * (class_weight * loss).mean(-1)
    return loss


def one_hot(array, bins):
    gt_labels = torch.arange(0, bins, device=array.device)
    gt_labels = gt_labels.unsqueeze(-1).unsqueeze(-1)
    mask = array.unsqueeze(0).type_as(gt_labels)
    mask = torch.eq(mask, gt_labels).float()

    return mask
