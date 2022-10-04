from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from detectron2.layers import Conv2d, get_norm

from .ops.modules import MSDeformAttn

from .ms_fpn import PositionEmbeddingSine, c2_xavier_fill, _get_activation_fn, _get_clones


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dims=256,
        feed_dims=1024,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()

        encoder_layer = MSDeformAttnTransformerEncoderLayer(embed_dims, feed_dims, num_heads, num_feature_levels,
                                                            enc_n_points, dropout, activation)
        self.layers = _get_clones(encoder_layer, num_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        output = src_flatten
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, lvl_pos_embed_flatten, reference_points, spatial_shapes, level_start_index,
                           mask_flatten)
        memory = output

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 feed_dims=1024,
                 num_heads=8,
                 num_levels=4,
                 n_points=4,
                 dropout=0.1,
                 activation="relu"):

        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(embed_dims, num_levels, num_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dims)

        # ffn
        self.linear1 = nn.Linear(embed_dims, feed_dims)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_dims, embed_dims)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dims)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MSDeformAttnHead(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        transformer_dropout: float,
        transformer_num_heads: int,
        transformer_feed_dims: int,
        transformer_layers: int,
        conv_dims: int,
        mask_dims: int,
        # deformable transformer encoder args
        transformer_in_channels: List[int],
        norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()
        # this is the input shape of pixel decoder
        self.in_channels = in_channels

        # this is the input shape of transformer encoder (could use less features than pixel decoder
        # starting from "res2" to "res5"
        self.transformer_in_channels = transformer_in_channels

        if len(self.transformer_in_channels) > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dims, kernel_size=1),
                        nn.GroupNorm(32, conv_dims),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dims, kernel_size=1),
                    nn.GroupNorm(32, conv_dims),
                )
            ])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoder(
            embed_dims=conv_dims,
            feed_dims=transformer_feed_dims,
            num_heads=transformer_num_heads,
            num_layers=transformer_layers,
            dropout=transformer_dropout,
            num_feature_levels=len(self.transformer_in_channels),
        )
        N_steps = conv_dims // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dims = mask_dims
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dims,
            mask_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        c2_xavier_fill(self.mask_features)

        # extra fpn levels
        self.num_fpn_levels = 4 - len(self.transformer_in_channels)

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.in_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dims)
            output_norm = get_norm(norm, conv_dims)

            lateral_conv = Conv2d(in_channels, conv_dims, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(
                conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, features):
        srcs = []
        pos = []
        # NOTE: self.in_channels is los to high resolution order.
        # So, we need to reverse feature maps into top-down order (from low to high resolution)
        sub_features = features[::-1]
        transformer_in_channels = self.transformer_in_channels[::-1]
        for idx, _ in enumerate(transformer_in_channels):
            x = sub_features[idx].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * len(self.transformer_in_channels)
        for i in range(len(self.transformer_in_channels)):
            if i < len(self.transformer_in_channels) - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        sub_features = features[:self.num_fpn_levels][::-1]
        in_channels = self.in_channels[:self.num_fpn_levels][::-1]
        for idx, _ in enumerate(in_channels):
            x = features[idx].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for idx, o in enumerate(out):
            if idx < len(self.transformer_in_channels):
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), multi_scale_features
