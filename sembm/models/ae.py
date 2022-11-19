import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.resnet38d import ResNet38d
from .backbones.vgg16d import VGG16d
from .backbones.resnets import ResNet101, ResNet50

# modules
from .mods import ASPP, PAMR, StochasticGate, GCI
from .backbones.base_net import BaseNet
from .utils import focal_loss, pseudo_gtmask, balanced_mask_loss_ce, resize
from ..apis.eval import augmentation, reverse_augmentation


class AE(BaseNet):

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', num_classes=21):
        super().__init__()

        if norm_type == 'syncbn':
            self.norm_layer = nn.SyncBatchNorm
        else:
            self.norm_layer = nn.BatchNorm2d

        self._init_backbone(backbone_name, pre_weights_path)  # initialise backbone weights

        # Decoder
        self._init_aspp()
        self._init_decoder(num_classes)

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
            weights_dict = torch.load(pre_weights_path)
            self.backbone.load_state_dict(weights_dict, False)

        self._lr_mult = self.backbone._lr_mult

        self.not_training = self.not_training + self.backbone.not_training
        self.bn_frozen = self.bn_frozen + self.backbone.bn_frozen
        self.from_scratch_layers = self.from_scratch_layers + self.backbone.from_scratch_layers

        self._fix_running_stats(self.backbone, fix_params=True)  # freeze backbone BNs

    def _init_aspp(self):
        self.aspp = ASPP(self.backbone.fan_out(), 8, self.norm_layer, 0.5)

        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, self.norm_layer):
                self.from_scratch_layers.append(m)

        self._fix_running_stats(self.aspp)  # freeze backbone BNs

    def _init_decoder(self, num_classes):

        self._aff = PAMR(10, [1, 2, 4, 8, 12, 24])

        def conv2d(*args, **kwargs):
            conv = nn.Conv2d(*args, **kwargs)
            self.from_scratch_layers.append(conv)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        def bnorm(*args, **kwargs):
            bn = self.norm_layer(*args, **kwargs)
            self.from_scratch_layers.append(bn)
            if bn.weight is not None:
                bn.weight.data.fill_(1)
                bn.bias.data.zero_()
            return bn

        # pre-processing for shallow features
        self.shallow_mask = GCI(self.norm_layer)
        self.from_scratch_layers += self.shallow_mask.from_scratch_layers

        # Stochastic Gate
        self.sg = StochasticGate()
        self.fc8_skip = nn.Sequential(conv2d(256, 48, 1, bias=False), bnorm(48), nn.ReLU())
        self.fc8_x = nn.Sequential(
            conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False), bnorm(256), nn.ReLU())

        # decoder
        self.cam_decoder = nn.Sequential(
            conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), bnorm(256), nn.ReLU(), nn.Dropout(0.5),
            conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), bnorm(256), nn.ReLU(), nn.Dropout(0.1),
            conv2d(256, num_classes - 1, kernel_size=1, stride=1))

    def run_pamr(self, im, mask):
        im = resize(im, mask.size()[-2:])
        masks_dec = self._aff(im, mask)
        return masks_dec

    def forward_backbone(self, x):
        res = self.backbone.forward_as_dict(x)
        res = list(res.values())
        return res[-1], res[1]

    def pix_encode(self, img):
        bottom_feats, top_feats = self.forward_backbone(img)
        aspp_feats = self.aspp(bottom_feats)
        skip_feats = self.fc8_skip(top_feats)
        up_feats = resize(aspp_feats, skip_feats.shape[-2:])
        x = self.fc8_x(torch.cat([up_feats, skip_feats], 1))

        x2 = self.shallow_mask(top_feats, x)
        pix_feats = self.sg(x, x2, alpha_rate=0.3)

        return pix_feats

    def inference(self, img):
        pix_feats = self.pix_encode(img)
        cams = self.cam_decoder(pix_feats)

        # constant BG scores
        bg = torch.ones_like(cams[:, :1])
        cams = torch.cat([bg, cams], dim=1)

        cam_probs = F.softmax(cams, dim=1)  # B x (C+1) x H x W

        # classification loss
        img_logits_1 = (cams * cam_probs).sum((-2, -1)) / (1.0 + cam_probs.sum((-2, -1)))
        # focal penalty loss
        img_logits_2 = focal_loss(cam_probs.mean((-2, -1)), p=3, c=0.01)
        # adding the losses together
        img_logits = img_logits_1[:, 1:] + img_logits_2[:, 1:]

        return img_logits, cams, cam_probs

    def tta_inference(self, batched_input, gt_label_filter, scales=[1.0], flip_directions=['none']):
        raw_img = batched_input['raw_img'].cuda()
        img = batched_input['img'].cuda()
        img_gt = batched_input['img_gt'].cuda()

        H, W = raw_img.shape[-2:]
        pix_preds = []
        for scale in scales:
            for flip_direction in flip_directions:
                simg = augmentation(img, scale, flip_direction, (H, W))

                img_logits, pix_logits, pix_probs = self.inference(simg)

                if gt_label_filter is False:
                    img_sigmoid = torch.sigmoid(img_logits)
                    img_gt = (img_sigmoid > 0.3)

                Cc = img_gt.shape[1]
                Cs = pix_logits.shape[1]
                pix_logits[:, (Cs - Cc):] = pix_logits[:, (Cs - Cc):] * img_gt[:, :, None, None]
                pix_probs[:, (Cs - Cc):] = pix_probs[:, (Cs - Cc):] * img_gt[:, :, None, None]

                pix_logits = reverse_augmentation(pix_logits, scale, flip_direction, (H, W))
                pix_probs = reverse_augmentation(pix_probs, scale, flip_direction, (H, W))

                pix_preds.append(pix_probs)

        pix_pred = sum(pix_preds) / len(pix_preds)

        # pix_pred = pix_pred[:, 1:]
        # # normalize cam
        # min_val = torch.min(pix_pred.flatten(2), dim=-1)[0][:, :, None, None]
        # max_val = torch.max(pix_pred.flatten(2), dim=-1)[0][:, :, None, None]
        # pix_pred = torch.nan_to_num((pix_pred - min_val) / (max_val - min_val + 1e-8), 0.0)

        # # set background threshold
        # bg = 1 - torch.max(pix_pred, dim=1, keepdim=True)[0]
        # pix_pred = torch.cat([bg, pix_pred], dim=1)
        # # bg = torch.ones_like(pix_pred[:, :1]) * 0.4
        # # pix_pred = torch.cat([bg, pix_pred], 1)

        # scale background by alpha power
        pix_pred[:, 0, ::] = torch.pow(pix_pred[:, 0, ::], 3)

        return pix_pred

    def forward(self, batched_inputs):
        img = batched_inputs['img']

        if self.training:
            raw_img = batched_inputs['raw_img']
            img_gt = batched_inputs['img_gt']
            img_logits, pix_logits, pix_probs = self.inference(img)

            losses = {}
            losses['loss_mlcls'] = F.multilabel_soft_margin_loss(img_logits, img_gt)

            if not batched_inputs['PRETRAIN']:
                # mask refinement with PAMR
                pix_probs_dec = self.run_pamr(raw_img, pix_probs.detach())
                # upscale the masks & clean
                pix_probs_dec = resize(pix_probs_dec, img.shape[-2:])
                pix_probs_dec[:, 1:] = pix_probs_dec[:, 1:] * img_gt[:, :, None, None]
                # create pseudo GT
                pseudo_gt_onehot = pseudo_gtmask(pix_probs_dec).detach()
                pix_logits = resize(pix_logits, img.shape[-2:])

                loss_cam_mask = balanced_mask_loss_ce(pix_logits, pseudo_gt_onehot, img_gt)

                losses['loss_cam_mask'] = loss_cam_mask

            return losses
        else:
            return self.tta_inference(batched_inputs, batched_inputs['gt_label_filter'], batched_inputs['scales'],
                                      batched_inputs['flip_directions'])
