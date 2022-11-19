import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.base_net import BaseNet
from .deeplabv1_seam import Deeplabv1SEAM
from .utils import resize
from ..apis.eval import augmentation, reverse_augmentation
from ..utils.distributed import all_reduce, get_world_size


class Deeplabv1SEAMDual(BaseNet):

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', num_classes=21):
        super().__init__()

        self.encoder_q = Deeplabv1SEAM(backbone_name, pre_weights_path, norm_type, num_classes)
        self.encoder_k = Deeplabv1SEAM(backbone_name, pre_weights_path, norm_type, num_classes)

        self.encoder_q.load_state_dict(
            torch.load(
                './work_dirs/voc_seam_pseudo_gt/deeplabv1_seam_sgd-lr7e-4_e30_bs16/epoch005_iter0003305_score0.625.pth')
            ['model'])

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.from_scratch_layers += self.encoder_q.from_scratch_layers

        self.ema_momentum = 0.99
        self.center_momentum = 0.99
        self.taut = 0.04
        self.taus = 0.1

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.ema_momentum + param_q.data * (1. - self.ema_momentum)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def _lr_mult(self):
        return 1., 2., 10., 20.

    def inference(self, img, encoder):
        x = encoder.backbone.forward(img)
        feature = encoder.conv_fov(x)
        feature = encoder.bn_fov(feature)
        feature = F.relu(feature, inplace=True)
        feature = encoder.conv_fov2(feature)
        feature = encoder.bn_fov2(feature)
        feature = F.relu(feature, inplace=True)
        feature = encoder.dropout1(feature)
        x_seg = encoder.cls_conv(feature)

        x_seg = resize(x_seg, img.shape[-2:], align_corners=False)

        return x_seg

    def tta_inference(self, batched_input, scales=[1.0], flip_directions=['none']):
        raw_img = batched_input['raw_img'].cuda()
        img = batched_input['img'].cuda()
        H, W = raw_img.shape[-2:]
        pix_preds = []

        for scale in scales:
            for flip_direction in flip_directions:

                simg = augmentation(img, scale, flip_direction, (H, W))
                pix_logits = self.inference(simg, self.encoder_q)
                pix_logits = reverse_augmentation(pix_logits, scale, flip_direction, (H, W))
                pix_preds.append(pix_logits)

        pix_pred = sum(pix_preds) / len(pix_preds)

        return pix_pred

    def forward(self, batched_inputs):
        img = batched_inputs['img']
        img_gt = batched_inputs['img_gt']
        pseudo_pix_gt = batched_inputs['pseudo_pix_gt']
        # multi-crop strategy
        pix_logits_q = self.inference(img, self.encoder_q)

        # compute key features
        with torch.no_grad():  # no gradient
            B, _, H, W = img.shape
            img_05 = F.interpolate(img, size=(int(H * 0.5), int(W * 0.5)), mode='bilinear', align_corners=False)
            img_15 = F.interpolate(img, size=(int(H * 1.5), int(W * 1.5)), mode='bilinear', align_corners=False)
            img_20 = F.interpolate(img, size=(int(H * 2), int(W * 2)), mode='bilinear', align_corners=False)
            pix_logits_k_05 = self.inference(img_05, self.encoder_k)
            pix_logits_k = self.inference(img, self.encoder_k)
            pix_logits_k_15 = self.inference(img_15, self.encoder_k)
            pix_logits_k_20 = self.inference(img_20, self.encoder_k)

            Hs, Ws = pix_logits_k.shape[-2:]

            pix_logits_k_05 = F.interpolate(pix_logits_k_05, size=(Hs, Ws), mode='bilinear', align_corners=False)
            pix_logits_k_15 = F.interpolate(pix_logits_k_15, size=(Hs, Ws), mode='bilinear', align_corners=False)
            pix_logits_k_20 = F.interpolate(pix_logits_k_20, size=(Hs, Ws), mode='bilinear', align_corners=False)

            pix_logits_k = sum([pix_logits_k_05, pix_logits_k, pix_logits_k_15, pix_logits_k_20]) / 4
            pix_logits_k[:, 1:] = pix_logits_k[:, 1:] * img_gt[:, :, None, None]

        _k = pix_logits_k.permute(0, 2, 3, 1)
        score, _ = torch.max(torch.softmax(_k, dim=-1), dim=-1)
        print(score)
        pseudo_k = torch.argmax(pix_logits_k, dim=1)
        pseudo_k[score < 0.6] = 255
        pseudo_k = pseudo_k[0].cpu().numpy()
        pix_gt = batched_inputs['pix_gt']
        raw_img = batched_inputs['raw_img']
        print(batched_inputs['filename'][0])
        import matplotlib.pyplot as plt
        import numpy as np
        plt.subplot(231)
        plt.imshow(raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        plt.subplot(232)
        plt.imshow(batched_inputs['pix_gt'][0].cpu().numpy())
        plt.subplot(233)
        plt.imshow(pseudo_pix_gt[0].cpu().numpy())
        plt.subplot(234)
        plt.imshow(torch.argmax(pix_logits_q, dim=1)[0].cpu().numpy())
        plt.subplot(235)
        plt.imshow(torch.argmax(pix_logits_k, dim=1)[0].cpu().numpy())
        plt.subplot(236)
        plt.imshow(score[0].cpu().numpy() > 0.9)
        plt.savefig(f'uncertain_train/{batched_inputs["filename"][0]}.png')

        if self.training:
            img = batched_inputs['img']
            img_s = batched_inputs['img_s']
            img_gt = batched_inputs['img_gt']
            pseudo_pix_gt = batched_inputs['pseudo_pix_gt']
            # multi-crop strategy
            pix_logits_q = self.inference(img_s, self.encoder_q)

            # compute key features
            with torch.no_grad():  # no gradient
                self._momentum_update_key_encoder()  # update the momentum encoder
                B, _, H, W = img.shape
                img_05 = F.interpolate(img, size=(int(H * 0.5), int(W * 0.5)), mode='bilinear', align_corners=False)
                img_15 = F.interpolate(img, size=(int(H * 1.5), int(W * 1.5)), mode='bilinear', align_corners=False)
                img_20 = F.interpolate(img, size=(int(H * 2), int(W * 2)), mode='bilinear', align_corners=False)
                pix_logits_k_05 = self.inference(img_05, self.encoder_k)
                pix_logits_k = self.inference(img, self.encoder_k)
                pix_logits_k_15 = self.inference(img_15, self.encoder_k)
                pix_logits_k_20 = self.inference(img_20, self.encoder_k)

                Hs, Ws = pix_logits_k.shape[-2:]

                pix_logits_k_05 = F.interpolate(pix_logits_k_05, size=(Hs, Ws), mode='bilinear', align_corners=False)
                pix_logits_k_15 = F.interpolate(pix_logits_k_15, size=(Hs, Ws), mode='bilinear', align_corners=False)
                pix_logits_k_20 = F.interpolate(pix_logits_k_20, size=(Hs, Ws), mode='bilinear', align_corners=False)

                pix_logits_k = sum([pix_logits_k_05, pix_logits_k, pix_logits_k_15, pix_logits_k_20]) / 4
                pix_logits_k[:, 1:] = pix_logits_k[:, 1:] * img_gt[:, :, None, None]

            # prob_k = F.softmax(pix_logits_k / self.taut, dim=1)
            # prob_k = torch.clamp(prob_k, 1e-3, 1 - 1e-3)
            # mask = torch.amax(prob_k, dim=(-3, -2, -1), keepdim=True).ge(0.8).float()
            # log_prob_k = prob_k.log()
            # prob_q = torch.log_softmax(pix_logits_q / self.taus, dim=1)

            # raw_img = batched_inputs['raw_img']
            # pix_pred_q = torch.argmax(pix_logits_q, dim=1)
            # pix_pred_k = torch.argmax(pix_logits_k, dim=1)
            # import matplotlib.pyplot as plt
            # import numpy as np
            # plt.subplot(131)
            # plt.imshow(raw_img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            # plt.subplot(132)
            # plt.imshow(pix_pred_q[0].cpu().numpy())
            # plt.subplot(133)
            # plt.imshow(pix_pred_k[0].cpu().numpy())
            # plt.savefig('2.png')
            # exit(0)

            losses = {}

            losses['loss_mask'] = F.cross_entropy(pix_logits_q, pseudo_pix_gt.long(), ignore_index=255).mean()
            # losses['loss_cons_mask'] = (F.kl_div(log_prob_k, prob_q, reduction='none') * mask).mean()

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
