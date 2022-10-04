import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.base_net import BaseNet
from .backbones.vgg16d import VGG16d
from .backbones.resnet38d import ResNet38d


class SimpleBaseline(BaseNet):
    """This baseline is acquired from PSA repo (https://github.com/jiwoon-ahn/psa)."""

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', num_classes=21):
        super().__init__()
        if norm_type == 'syncbn':
            self.norm_layer = nn.SyncBatchNorm
        else:
            self.norm_layer = nn.BatchNorm2d

        self.backbone_name = backbone_name
        self._init_backbone(backbone_name, pre_weights_path)  # initialise backbone weights

        self.dropout = nn.Dropout2d(p=0.5)
        self.cam_decoder = nn.Conv2d(self.backbone.fan_out(), num_classes - 1, 1, bias=False)

    def _init_backbone(self, backbone_name, pre_weights_path):
        if backbone_name == "resnet38":
            print("Backbone: ResNet38")
            self.backbone = ResNet38d(norm_layer=self.norm_layer)
        elif backbone_name == "vgg16":
            print("Backbone: VGG16")
            self.backbone = VGG16d()
        else:
            raise NotImplementedError("No backbone found for '{}'".format(backbone_name))

        if pre_weights_path is not None:
            print("Loading backbone weights from: ", pre_weights_path)
            weights_dict = torch.load(pre_weights_path)
            self.backbone.load_state_dict(weights_dict, False)

        self._lr_mult = self.backbone._lr_mult

        self.not_training = self.not_training + self.backbone.not_training
        self.bn_frozen = self.bn_frozen + self.backbone.bn_frozen
        self.from_scratch_layers = self.from_scratch_layers + self.backbone.from_scratch_layers

        self._fix_running_stats(self.backbone, fix_params=True)  # freeze backbone BNs

    def forward(self, batched_inputs):
        img = batched_inputs['img']
        raw_img = batched_inputs['raw_img']
        img_gt = batched_inputs['img_gt']

        # NOTE: convert to AffinityNet style norm
        if self.backbone_name == 'vgg16':
            img = raw_img[:, (2, 1, 0)]
            img[:, 0] = img[:, 0] - 104.008
            img[:, 1] = img[:, 1] - 116.669
            img[:, 2] = img[:, 2] - 122.675

        x = self.backbone(img)
        x = self.dropout(x)
        pix_logits = self.cam_decoder(x)

        img_logits = F.avg_pool2d(pix_logits, kernel_size=(x.size(2), x.size(3)), padding=0)
        img_logits = img_logits.view(-1, 20)

        if self.training:
            return {'loss_mlcls': F.multilabel_soft_margin_loss(x, img_gt)}
        else:
            pix_logits_cam = torch.sqrt(F.relu(pix_logits))
            pix_logits_cam = pix_logits_cam / (
                torch.max(pix_logits_cam.flatten(-2), dim=-1)[0][:, :, None, None] + 1e-5)

            bg = torch.ones_like(pix_logits_cam[:, :1]) * 0.2
            pix_logits_cam = torch.cat([bg, pix_logits_cam], 1)

            pix_logits_cam = F.interpolate(pix_logits_cam, img.shape[-2:], mode='bilinear', align_corners=False)

            return {
                'img_logits': img_logits,
                'pix_logits_cam': pix_logits_cam,
            }
