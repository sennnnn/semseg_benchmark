from threading import local
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.vgg16d import VGG16d
from .backbones.resnets import ResNet101, ResNet50
from .backbones.base_net import BaseNet
from .resnet38d import ResNet38d
from .utils import resize
from ..apis.eval import augmentation, reverse_augmentation


class Deeplabv1CAN(BaseNet):

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', num_classes=21):
        super().__init__()

        if norm_type == 'syncbn':
            self.norm_layer = nn.SyncBatchNorm
        else:
            self.norm_layer = nn.BatchNorm2d

        self._init_backbone(backbone_name, pre_weights_path)

        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

        self.from_scratch_layers = [self.fc8_seg_conv1, self.fc8_seg_conv2]

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

        if hasattr(self.backbone, '_lr_mult'):
            self._lr_mult = self.backbone._lr_mult

        if hasattr(self.backbone, 'not_training'):
            self.not_training = self.not_training + self.backbone.not_training

        if hasattr(self.backbone, 'bn_frozen'):
            self.bn_frozen = self.bn_frozen + self.backbone.bn_frozen

        if hasattr(self.backbone, 'from_scratch_layers'):
            self.from_scratch_layers = self.from_scratch_layers + self.backbone.from_scratch_layers

        self._fix_running_stats(self.backbone, fix_params=True)  # freeze backbone BNs

    def _lr_mult(self):
        return 1., 1., 10., 10.

    def inference(self, img):
        x = self.backbone.forward(img)
        x_seg = F.relu(self.fc8_seg_conv1(x))
        x_seg = self.fc8_seg_conv2(x_seg)

        x_seg = resize(x_seg, img.shape[-2:], align_corners=False)

        return x_seg

    def tta_inference(self, batched_input, scales=[1.0], flip_directions=['none']):
        raw_img = batched_input['raw_img'].cuda()
        img = batched_input['img'].cuda()
        img_gt = batched_input['img_gt'].cuda()
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
        pix_gt = batched_inputs['pix_gt']
        pseudo_pix_gt = batched_inputs['pseudo_pix_gt']

        if self.training:
            pix_logits = self.inference(img)
            losses = {}
            losses['loss_mask'] = F.cross_entropy(pix_logits, pseudo_pix_gt.long(), ignore_index=255).mean()

            pix_preds = torch.argmax(pix_logits.detach(), dim=1)
            # including logits
            can_loss = 0
            for pix_logit, pix_pred, ig, ppg, pg in zip(pix_logits, pix_preds, img_gt, pseudo_pix_gt, pix_gt):
                class_ids = torch.nonzero(ig)[:, 0] + 1
                reverse_class_ids = torch.nonzero((1 - ig))[:, 0] + 1

                pred_class_ids = torch.unique(pix_pred)
                pred_class_ids = pred_class_ids[pred_class_ids != 0]
                # print(pred_class_ids, class_ids)

                # pix_pred
                # print(class_ids, reverse_class_ids)
                in_logit = pix_logit[class_ids]
                in_logit = torch.cat([pix_logit[:1], in_logit], dim=0)
                out_logit = pix_logit[reverse_class_ids]

                # print(pix_logit[0, 23, 23])
                # print(in_logit[:, 23, 23])
                # print(out_logit[:, 23, 23])
                in_logit, _ = torch.topk(in_logit, k=1, dim=0)
                out_logit, _ = torch.topk(out_logit, k=1, dim=0)
                # print(out_logit[:, 23, 23])
                # print(pg[23, 23], ppg[23, 23])
                local_loss = nn.Softplus()(torch.logsumexp(-in_logit, dim=0) + torch.logsumexp(out_logit, dim=0))
                # local_loss_ = torch.log(1 + 1e-5 +
                #                         torch.sum(torch.exp(-in_logit), dim=0) * torch.sum(torch.exp(out_logit), dim=0))
                # print(local_loss.mean(), local_loss_.mean())
                # exit(0)
                # local_loss = local_loss[pix_pred != 0]
                local_loss = torch.sum(local_loss) if len(local_loss) == 0 else local_loss.mean()
                # if torch.isnan(local_loss):
                #     print(
                #         torch.log(1 + 1e-5 +
                #                   torch.sum(torch.exp(-in_logit), dim=0) * torch.sum(torch.exp(out_logit), dim=0)))
                #     print(
                #         torch.log(1 + 1e-5 + torch.sum(torch.exp(-in_logit), dim=0) *
                #                   torch.sum(torch.exp(out_logit), dim=0))[pg != 0].mean())
                can_loss += local_loss

            losses['loss_can_mask'] = can_loss / img_gt.shape[0]

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
