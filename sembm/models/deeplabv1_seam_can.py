import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.vgg16d import VGG16d
from .backbones.resnets import ResNet101, ResNet50
from .backbones.base_net import BaseNet
from .resnet38d import ResNet38d
from .utils import resize
from ..apis.eval import augmentation, reverse_augmentation


def can_loss(pix_logit, img_gt):
    class_ids = torch.nonzero(img_gt)[:, 0] + 1
    reverse_class_ids = torch.nonzero((1 - img_gt))[:, 0] + 1

    pix_logit = pix_logit.permute(1, 2, 0).contiguous()
    in_logit = pix_logit[:, :, class_ids]
    in_logit = torch.cat([pix_logit[:, :, :1], in_logit], dim=-1)
    out_logit = pix_logit[:, :, reverse_class_ids]

    _, rank_ids = torch.sort(in_logit, dim=-1, descending=True)

    topk_mask = torch.zeros_like(in_logit)
    ones_mask = torch.ones_like(in_logit)
    for i in range(len(class_ids) + 1):
        mask = rank_ids[:, :, i] == 0
        sel_in_pixs = in_logit[mask]

        if i == 0:
            sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=1, dim=-1)
        else:
            sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=i, dim=-1)

        topk_mask[mask] = torch.scatter(topk_mask[mask], dim=-1, index=topk_ids, src=ones_mask[mask])
        # print(i, topk_ids[2], topk_mask[mask][2], sel_in_pixs.shape)

    # in_logit, _ = torch.topk(in_logit, k=1, dim=0)
    in_logit = in_logit * topk_mask
    out_logit, _ = torch.topk(out_logit, k=1, dim=-1)
    local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=-1) + torch.logsumexp(out_logit + 6, dim=-1))

    return local_loss

class Deeplabv1SEAMCAN(BaseNet):

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', num_classes=21, alpha=1.0, margin=0.0):
        super().__init__()

        self.alpha = alpha
        self.margin = margin

        if norm_type == 'syncbn':
            self.norm_layer = nn.SyncBatchNorm
        else:
            self.norm_layer = nn.BatchNorm2d

        self._init_backbone(backbone_name, pre_weights_path)

        self.conv_fov = nn.Conv2d(4096, 512, 3, 1, padding=12, dilation=12, bias=False)
        self.bn_fov = self.norm_layer(512, momentum=0.0003, affine=True)
        self.conv_fov2 = nn.Conv2d(512, 512, 1, 1, padding=0, bias=False)
        self.bn_fov2 = self.norm_layer(512, momentum=0.0003, affine=True)
        self.dropout1 = nn.Dropout(0.5)
        self.cls_conv = nn.Conv2d(512, num_classes, 1, 1, padding=0)

        self.from_scratch_layers = [self.conv_fov, self.conv_fov2, self.cls_conv]

    def _init_backbone(self, backbone_name, pre_weights_path):

        if backbone_name == "resnet38d":
            print("Backbone: ResNet38")
            self.backbone = ResNet38d(norm_layer=self.norm_layer)
        elif backbone_name == "vgg16d":
            print("Backbone: VGG16")
            self.backbone = VGG16d()
        elif backbone_name == "resnet50":
            print("Backbone: ResNet50")
            self.backbone = ResNet50(norm_layer=self.norm_layer)
        elif backbone_name == "resnet101":
            print("Backbone: ResNet101")
            self.backbone = ResNet101(norm_layer=self.norm_layer)
        else:
            raise NotImplementedError("No backbone found for '{}'".format(backbone_name))

        if pre_weights_path is not None:
            print("Loading backbone weights from: ", pre_weights_path)
            weights_dict = torch.load(pre_weights_path, map_location='cpu')
            self.backbone.load_state_dict(weights_dict, False)

        if not hasattr(self, '_lr_mult'):
            if hasattr(self.backbone, '_lr_mult'):
                self._lr_mult = self.backbone._lr_mult

    def _lr_mult(self):
        return 1., 2., 10., 20.

    def inference(self, img):
        x = self.backbone.forward(img)
        feature = self.conv_fov(x)
        feature = self.bn_fov(feature)
        feature = F.relu(feature, inplace=True)
        feature = self.conv_fov2(feature)
        feature = self.bn_fov2(feature)
        feature = F.relu(feature, inplace=True)
        feature = self.dropout1(feature)
        x_seg = self.cls_conv(feature)

        x_seg = resize(x_seg, img.shape[-2:], align_corners=False)

        return x_seg

    def tta_inference(self, batched_input, scales=[1.0], flip_directions=['none']):
        raw_img = batched_input['raw_img'].cuda()
        img = batched_input['img'].cuda()
        # img_gt = batched_input['img_gt'].cuda()
        H, W = raw_img.shape[-2:]
        pix_preds = []

        for scale in scales:
            for flip_direction in flip_directions:

                simg = augmentation(img, scale, flip_direction, (H, W))
                pix_logits = self.inference(simg)
                # pix_logits[:, 1:, :, :] = pix_logits[:, 1:, :, :] * img_gt[:, :, None, None]
                pix_logits = reverse_augmentation(pix_logits, scale, flip_direction, (H, W))
                pix_preds.append(pix_logits)

        pix_pred = sum(pix_preds) / len(pix_preds)
        return pix_pred

    def forward(self, batched_inputs):
        img = batched_inputs['img']
        img_gt = batched_inputs['img_gt']
        pseudo_pix_gt = batched_inputs['pseudo_pix_gt']

        if self.training:
            pix_logits = self.inference(img)
            losses = {}

            # including logits (top1, top1)
            # can_loss = 0
            # for pix_logit, ig in zip(pix_logits, img_gt):
            #     class_ids = torch.nonzero(ig)[:, 0] + 1
            #     reverse_class_ids = torch.nonzero((1 - ig))[:, 0] + 1

            #     in_logit = pix_logit[class_ids]
            #     in_logit = torch.cat([pix_logit[:1], in_logit], dim=0)
            #     out_logit = pix_logit[reverse_class_ids]

            #     in_logit, _ = torch.topk(in_logit, k=1, dim=0)
            #     out_logit, _ = torch.topk(out_logit, k=1, dim=0)
            #     local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=0) + torch.logsumexp(out_logit, dim=0))

            #     can_loss += local_loss.mean()

            # can_loss = 0
            # for pix_logit, ig in zip(pix_logits, img_gt):
            #     class_ids = torch.nonzero(ig)[:, 0] + 1
            #     reverse_class_ids = torch.nonzero((1 - ig))[:, 0] + 1

            #     in_logit = pix_logit[class_ids]
            #     in_logit = torch.cat([pix_logit[:1], in_logit], dim=0)
            #     out_logit = pix_logit[reverse_class_ids]

            #     local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=0) + torch.logsumexp(out_logit, dim=0))

            #     can_loss += local_loss.mean()

            # can_loss = 0
            # for pix_logit, ig in zip(pix_logits, img_gt):
            #     class_ids = torch.nonzero(ig)[:, 0] + 1
            #     reverse_class_ids = torch.nonzero((1 - ig))[:, 0] + 1

            #     pix_logit = pix_logit.permute(1, 2, 0).contiguous()
            #     in_logit = pix_logit[:, :, class_ids]
            #     in_logit = torch.cat([pix_logit[:, :, :1], in_logit], dim=-1)
            #     out_logit = pix_logit[:, :, reverse_class_ids]

            #     _, rank_ids = torch.sort(in_logit, dim=-1, descending=True)

            #     local_loss = []
            #     topk_mask = torch.zeros_like(in_logit)
            #     ones_mask = torch.ones_like(in_logit)
            #     for i in range(len(class_ids) + 1):
            #         mask = rank_ids[:, :, i] == 0
            #         sel_in_pixs = in_logit[mask]

            #         if i == 0:
            #             sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=1, dim=-1)
            #         else:
            #             sel_in_pixs, topk_ids = torch.topk(sel_in_pixs, k=i, dim=-1)

            #         topk_mask[mask] = torch.scatter(topk_mask[mask], dim=-1, index=topk_ids, src=ones_mask[mask])
            #         # print(i, topk_ids[2], topk_mask[mask][2], sel_in_pixs.shape)

            #     # in_logit, _ = torch.topk(in_logit, k=1, dim=0)
            #     in_logit = in_logit * topk_mask
            #     out_logit, _ = torch.topk(out_logit, k=1, dim=-1)
            #     local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=-1) +
            #                                torch.logsumexp(out_logit + self.margin, dim=-1)).mean()
            #     can_loss += local_loss.mean()

            losses['loss_mask'] = F.cross_entropy(pix_logits, pseudo_pix_gt.long(), ignore_index=255).mean()
            losses['loss_can_mask'] = self.alpha * can_loss / img_gt.shape[0]

            # if batched_inputs['PRETRAIN']:
            #     losses['loss_mask'] = F.cross_entropy(pix_logits, pseudo_pix_gt.long(), ignore_index=255).mean()
            # else:
            #     # NOTE: filter out wrong pixels
            #     # pix_preds = torch.argmax(pix_logits.detach(), dim=1)
            #     # clean_pix_logits = pix_logits.detach().clone()
            #     # clean_pix_logits[:, 1:] = clean_pix_logits[:, 1:] * img_gt[:, :, None, None]
            #     # clean_pix_preds = torch.argmax(clean_pix_logits, dim=1)

            #     loss_mask = F.cross_entropy(pix_logits, pseudo_pix_gt.long(), ignore_index=255, reduction='none')
            #     losses['loss_mask'] = torch.sum(loss_mask) if len(loss_mask) == 0 else loss_mask.mean()

            #     # loss_mask = F.cross_entropy(
            #     #     pix_logits, pseudo_pix_gt.long(), ignore_index=255, reduction='none')[clean_pix_preds == pix_preds]
            #     # losses['loss_mask'] = torch.sum(loss_mask) if len(loss_mask) == 0 else loss_mask.mean()
            #     # loss_clean_mask = F.cross_entropy(
            #     #     pix_logits, clean_pix_preds.long(), ignore_index=255, reduction='none')[clean_pix_preds != pix_preds]
            #     # losses['loss_clean_mask'] = torch.sum(loss_clean_mask) if len(
            #     #     loss_clean_mask) == 0 else loss_clean_mask.mean()
            #     # including logits
            #     can_loss = 0
            #     for pix_logit, ig in zip(pix_logits, img_gt):
            #         class_ids = torch.nonzero(ig)[:, 0] + 1
            #         reverse_class_ids = torch.nonzero((1 - ig))[:, 0] + 1

            #         # pred_class_ids = torch.unique(pix_pred)
            #         # pred_class_ids = pred_class_ids[pred_class_ids != 0]
            #         # pred_class_mask = torch.zeros_like(ig)
            #         # pred_class_mask[pred_class_ids - 1] = 1
            #         # pred_class_mask *= 1 - ig
            #         # abnormal_class_ids = torch.nonzero(pred_class_mask)[:, 0] + 1

            #         in_logit = pix_logit[class_ids]
            #         in_logit = torch.cat([pix_logit[:1], in_logit], dim=0)
            #         out_logit = pix_logit[reverse_class_ids]

            #         in_logit, _ = torch.topk(in_logit, k=1, dim=0)
            #         out_logit, _ = torch.topk(out_logit, k=1, dim=0)
            #         local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=0) + torch.logsumexp(out_logit, dim=0))

            #         can_loss += torch.sum(local_loss) if len(local_loss) == 0 else local_loss.mean()

            #     losses['loss_can_mask'] = can_loss / img_gt.shape[0]

            return losses
        else:
            return self.tta_inference(batched_inputs, batched_inputs['scales'], batched_inputs['flip_directions'])

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def parameter_groups(self, base_lr, wd, **kwargs):
        w_old, b_old, w_new, b_new = self._lr_mult()

        groups = (
            {
                "params": [],
                "weight_decay": wd,
                "lr": w_old * base_lr
            },  # weight learning
            {
                "params": [],
                "weight_decay": 0.0,
                "lr": b_old * base_lr
            },  # bias finetuning
            {
                "params": [],
                "weight_decay": wd,
                "lr": w_new * base_lr
            },  # weight finetuning
            {
                "params": [],
                "weight_decay": 0.0,
                "lr": b_new * base_lr
            })  # bias learning

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.SyncBatchNorm):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2]["params"].append(m.weight)
                    else:
                        groups[0]["params"].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[2]["params"].append(m.bias)
                    else:
                        groups[0]["params"].append(m.bias)

        for i, g in enumerate(groups):
            print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))

        return groups
