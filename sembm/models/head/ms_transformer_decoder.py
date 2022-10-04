from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ms_fpn import PositionEmbeddingSine, _get_activation_fn


class SelfAttentionLayer(nn.Module):

    def __init__(self, model_dims, num_heads, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dims, num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(model_dims)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt


class CrossAttentionLayer(nn.Module):

    def __init__(self, model_dims, num_heads, dropout=0.0, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(model_dims, num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(model_dims)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):

    def __init__(self, model_dims, feed_dims=2048, dropout=0.0, activation="relu"):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(model_dims, feed_dims)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_dims, model_dims)

        self.norm = nn.LayerNorm(model_dims)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleTransformerDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes: int,
        num_heads: int,
        feed_dims: int,
        hidden_dims: int,
        mask_dims: int,
        num_queries: int,
        num_layers: int,
    ):
        super().__init__()

        self.num_classes = num_classes

        # positional encoding
        N_steps = hidden_dims // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sa_layers = nn.ModuleList()
        self.ca_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.ca_layers.append(CrossAttentionLayer(
                model_dims=hidden_dims,
                num_heads=num_heads,
                dropout=0.0,
            ))

            self.sa_layers.append(SelfAttentionLayer(
                model_dims=hidden_dims,
                num_heads=num_heads,
                dropout=0.0,
            ))

            self.ffn_layers.append(FFNLayer(
                model_dims=hidden_dims,
                feed_dims=feed_dims,
                dropout=0.0,
            ))

        self.decoder_norm = nn.LayerNorm(hidden_dims)

        self.num_queries = num_queries
        # learnable query features
        self.query_obj = nn.Embedding(num_queries, hidden_dims)
        # learnable query p.e.
        self.query_pos = nn.Embedding(num_queries, hidden_dims)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dims)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dims:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dims, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dims, num_classes + 1)
        self.mask_embed = MLP(hidden_dims, hidden_dims, mask_dims, 3)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_obj.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.ca_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_pos)

            output = self.sa_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos)

            # FFN
            output = self.ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        return output, predictions_class[-1], predictions_mask[-1]

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) <
                     0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
