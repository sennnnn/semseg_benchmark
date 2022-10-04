import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from sswss.models.utils import resize
from sswss.apis.eval import augmentation, reverse_augmentation

from .backbones.vision_transformer import VisionTransformer, _cfg, vit_small_patch16_224

__all__ = ['deit_small_MCTformerV1_patch16_224', 'deit_small_MCTformerV2_patch16_224']


class MCTformerV2(VisionTransformer):

    def __init__(self, backbone_name, pre_weights_path, norm_type='syncbn', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes -= 1
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self._init_backbone(pre_weights_path)

    def _init_backbone(self, pre_weights_path):
        if pre_weights_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(pre_weights_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(pre_weights_path, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint

        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.patch_embed.num_patches
        if pre_weights_path.startswith('https'):
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)

        new_size = int(num_patches**0.5)

        if pre_weights_path.startswith('https'):
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1, self.num_classes, 1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        if pre_weights_path.startswith('https'):
            cls_token_checkpoint = checkpoint_model['cls_token']
            new_cls_token = cls_token_checkpoint.repeat(1, self.num_classes, 1)
            checkpoint_model['cls_token'] = new_cls_token

        self.load_state_dict(checkpoint_model, strict=False)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights

    def inference(self, img, n_layers, attention_type):
        w, h = img.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(img)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])

        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = [torch.mean(x, dim=1) for x in attn_weights]
        sum_attn_weights = sum(attn_weights)

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:]
        mtatt = sum(mtatt)[:, 0:self.num_classes, self.num_classes:]
        mtatt = mtatt.reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt

        patch_attn = sum_attn_weights[:, self.num_classes:, self.num_classes:]
        x_cls_logits = x_cls.mean(-1)

        return x_cls_logits, x_patch_logits, cams, patch_attn

    def tta_inference(self, batched_input, gt_label_filter, scales=[1.0], flip_directions=['none']):
        raw_img = batched_input['raw_img'].cuda()
        img = batched_input['img'].cuda()
        img_gt = batched_input['img_gt'].cuda()

        H, W = raw_img.shape[-2:]
        img_preds = []
        pix_preds = []
        for scale in scales:
            for flip_direction in flip_directions:
                simg = augmentation(img, scale, flip_direction, (H, W))

                img_logits, patch_img_logits, cams, patch_attn = self.inference(
                    simg, n_layers=3, attention_type='fused')

                h_featmap = simg.shape[-2] // 16
                w_featmap = simg.shape[-1] // 16

                cams = torch.einsum('bnm,bcm->bcn', patch_attn, cams.flatten(2))
                cams = cams.reshape(cams.shape[0], cams.shape[1], h_featmap, w_featmap)
                cams = resize(cams, img.shape[-2:], align_corners=False)

                if gt_label_filter is False:
                    img_sigmoid = torch.sigmoid(img_logits)
                    img_gt = (img_sigmoid > 0.3)
                cams = cams * img_gt[:, :, None, None]

                # # normalize cam
                # min_val = torch.min(cams.flatten(2), dim=-1)[0][:, :, None, None]
                # max_val = torch.max(cams.flatten(2), dim=-1)[0][:, :, None, None]
                # cams = torch.nan_to_num((cams - min_val) / (max_val - min_val + 1e-8), 0.0)

                # # set background threshold
                # bg = 1 - torch.max(cams, dim=1, keepdim=True)[0]
                # cams = torch.cat([bg, cams], dim=1)

                cams = reverse_augmentation(cams, scale, flip_direction, (H, W))

                img_preds.append(img_logits)
                pix_preds.append(cams)

        img_pred = sum(img_preds) / len(img_preds)
        pix_pred = sum(pix_preds) / len(pix_preds)

        # normalize cam
        min_val = torch.min(pix_pred.flatten(2), dim=-1)[0][:, :, None, None]
        max_val = torch.max(pix_pred.flatten(2), dim=-1)[0][:, :, None, None]
        pix_pred = torch.nan_to_num((pix_pred - min_val) / (max_val - min_val + 1e-8), 0.0)

        # set background threshold
        bg = 1 - torch.max(pix_pred, dim=1, keepdim=True)[0]
        pix_pred = torch.cat([bg, pix_pred], dim=1)
        # bg = torch.ones_like(pix_pred[:, :1]) * 0.4
        # pix_pred = torch.cat([bg, pix_pred], 1)

        # scale background by alpha power
        pix_pred[:, 0, ::] = torch.pow(pix_pred[:, 0, ::], 2)

        return img_pred, pix_pred

    def forward(self, batched_input):
        raw_img = batched_input['raw_img']
        img = batched_input['img']
        img_gt = batched_input['img_gt']
        n_layers = 3
        attention_type = 'fused'

        if self.training:
            img_logits, patch_img_logits, cams, patch_attn = self.inference(img, n_layers, attention_type)

            losses = {}
            losses['loss_mlcls'] = F.multilabel_soft_margin_loss(img_logits, img_gt)
            losses['loss_patch_mlcls'] = F.multilabel_soft_margin_loss(patch_img_logits, img_gt)

            return losses

            # _mtatt = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/mtatt.npy')).cuda()
            # print(torch.allclose(mtatt, _mtatt))

            # _cams = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/cams.npy')).cuda()
            # print(torch.allclose(cams, _cams))

            # _patch_attn = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/patch_attn.npy')).cuda()
            # print(torch.allclose(patch_attn, _patch_attn))

            # _x_cls_logits = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/x_cls_logits.npy')).cuda()
            # print(torch.allclose(x_cls_logits, _x_cls_logits))
        else:
            return self.tta_inference(batched_input, batched_input['gt_label_filter'], batched_input['scales'],
                                      batched_input['flip_directions'])

        if not self.training:
            w, h = img.shape[2] - img.shape[2] % 16, img.shape[3] - img.shape[3] % 16
            w_featmap = w // 16
            h_featmap = h // 16

            cams = torch.einsum('bnm,bcm->bcn', patch_attn, cams.flatten(2))
            cams = cams.reshape(cams.shape[0], cams.shape[1], w_featmap, h_featmap)
            cams = resize(cams, img.shape[-2:], align_corners=False)
            cams = cams * img_gt[:, :, None, None]

            # final_cams = torch.tensor(
            #     np.load('../../reproduces/weak_sup_semseg/MCTformer/final_cls_attentions.npy')).cuda()
            # print(torch.allclose(cams, final_cams))
            # exit(0)

            min_val = torch.min(cams.flatten(2), dim=-1)[0][:, :, None, None]
            max_val = torch.max(cams.flatten(2), dim=-1)[0][:, :, None, None]
            cams = torch.nan_to_num((cams - min_val) / (max_val - min_val + 1e-8), 0.0)

            # cams_ = torch.tensor(
            #     np.load('../../reproduces/weak_sup_semseg/MCTformer/cls_attentions_0.npy')).cuda()
            # print(torch.allclose(cams[0, 0], cams_, 1e-4))

            bg = 1 - torch.max(cams, dim=1, keepdim=True)[0]
            pix_logits_cam = torch.cat([bg, cams], dim=1)

            # bg = torch.ones_like(pix_logits_cam[:, :1]) * 0.4
            # pix_logits_cam = torch.cat([bg, pix_logits_cam], 1)

            # import numpy as np
            # from sswss.utils.dcrf import crf_inference
            # bgcam_score = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/bgcam_score.npy')).cuda()
            # print(pix_logits_cam.shape, bgcam_score.shape)
            # print(torch.allclose(pix_logits_cam[0, 0], bgcam_score[0]))
            # # print(torch.allclose(pix_logits_cam[0, 1], bgcam_score[1]))
            # print(torch.allclose(pix_logits_cam[0, 20], bgcam_score[1]))

            # orig_img = raw_img[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
            # canvas = torch.zeros_like(pix_logits_cam)
            # img_gt = img_gt[0]
            # ids = [0]
            # for cls_id, val in enumerate(img_gt.long()):
            #     if val == 1:
            #         ids.append(cls_id + 1)

            # _orig_img = np.load('../../reproduces/weak_sup_semseg/MCTformer/orig_img.npy')

            # pix_logits_cam = pix_logits_cam[0, ids]
            # pix_logits_cam = crf_inference(_orig_img, pix_logits_cam.cpu().numpy(), labels=pix_logits_cam.shape[0])
            # pix_logits_cam = torch.tensor(pix_logits_cam).cuda()
            # canvas[0, ids] = pix_logits_cam
            # pix_logits_cam = canvas

            # crf_score = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/crf_score.npy')).cuda()
            # print(torch.allclose(crf_score[0], pix_logits_cam[0, 0]))
            # print(torch.allclose(crf_score[1], pix_logits_cam[0, 1]))
            # print(torch.allclose(crf_score[2], pix_logits_cam[0, 15]))

            # exit(0)

            return {
                'img_logits': img_logits,
                'pix_probs_cam': F.softmax(pix_logits_cam, dim=1),
                'pix_logits_cam': pix_logits_cam,
            }

            import numpy as np
            from sswss.utils.dcrf import crf_inference

            pix_logits_cam = pix_logits_cam[0]
            # bgcam_score = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/bgcam_score.npy')).cuda()
            # print(pix_logits_cam.shape, bgcam_score.shape)
            # print(torch.allclose(pix_logits_cam[0], bgcam_score[0]))
            # print(torch.allclose(pix_logits_cam[1], bgcam_score[1]))
            # print(torch.allclose(pix_logits_cam[15], bgcam_score[2]))

            # _orig_img = np.load('../../reproduces/weak_sup_semseg/MCTformer/orig_img.npy')

            orig_img = raw_img[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
            canvas = torch.zeros_like(pix_logits_cam)
            img_gt = img_gt[0]
            ids = [0]
            for cls_id, val in enumerate(img_gt.long()):
                if val == 1:
                    ids.append(cls_id + 1)

            pix_logits_cam = pix_logits_cam[ids]
            pix_logits_cam = crf_inference(orig_img, pix_logits_cam.cpu().numpy(), labels=pix_logits_cam.shape[0])
            pix_logits_cam = torch.tensor(pix_logits_cam).cuda()
            canvas[ids] = pix_logits_cam
            pix_logits_cam = canvas[None]

            # crf_score = torch.tensor(np.load('../../reproduces/weak_sup_semseg/MCTformer/crf_score.npy')).cuda()
            # print(torch.allclose(crf_score[0], pix_logits_cam[0, 0]))
            # print(torch.allclose(crf_score[1], pix_logits_cam[0, 1]))
            # print(torch.allclose(crf_score[2], pix_logits_cam[0, 15]))

            # import matplotlib.pyplot as plt
            # plt.subplot(221)
            # plt.imshow(orig_img)
            # plt.subplot(222)
            # plt.imshow(np.argmax(pix_logits_cam, axis=0))
            # print(np.unique(np.argmax(pix_logits_cam, axis=0)))
            # plt.subplot(223)
            # plt.imshow(_orig_img)
            # plt.subplot(224)
            # plt.imshow(np.argmax(crf_score, axis=0))
            # print(np.unique(np.argmax(crf_score, axis=0)))
            # plt.savefig('2.png')

            return {
                'img_logits': img_logits,
                'pix_probs_cam': F.softmax(pix_logits_cam, dim=1),
                'pix_logits_cam': pix_logits_cam
            }

    def parameter_groups(self, base_lr, wd, batch_size, world_size):
        base_lr = base_lr * batch_size * world_size / 512.0
        print(base_lr)
        exit(0)
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [{
            'params': no_decay,
            'weight_decay': 0.,
            "lr": base_lr
        }, {
            'params': decay,
            'weight_decay': wd,
            "lr": base_lr
        }]


class MCTformerV1(VisionTransformer):

    def __init__(self, last_opt='average', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        if last_opt == 'fc':
            self.head = nn.Conv1d(
                in_channels=self.num_classes,
                out_channels=self.num_classes,
                kernel_size=self.embed_dim,
                groups=self.num_classes)
            self.head.apply(self._init_weights)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights_i)
        return x[:, 0:self.num_classes], attn_weights

    def forward(self, x, n_layers=12, return_att=False):
        x, attn_weights = self.forward_features(x)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:]
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x.mean(-1)

        if return_att:
            return x_cls_logits, mtatt, patch_attn
        else:
            return x_cls_logits


@register_model
def deit_small_MCTformerV2_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV2(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True)['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True)
        model.load_state_dict(checkpoint["model"])

    return model
