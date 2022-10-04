from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ms_fpn import PositionEmbeddingSine, TransformerEncoder, _get_activation_fn, _get_clones


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dims, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dims] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dims=256,
        feed_dims=2048,
        num_heads=8,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dims, feed_dims)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_dims, embed_dims)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class Transformer(nn.Module):

    def __init__(
        self,
        embed_dims=512,
        feed_dims=2048,
        num_heads=8,
        num_dec_layers=6,
        num_enc_layers=6,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        self.encoder = TransformerEncoder(embed_dims, feed_dims, num_heads, num_enc_layers, dropout, activation)
        self.decoder = TransformerDecoder(embed_dims, feed_dims, num_heads, num_dec_layers, dropout, activation)

        self._reset_parameters()

        self.embed_dims = embed_dims
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerDecoder(nn.Module):

    def __init__(self,
                 embed_dims=512,
                 feed_dims=2048,
                 num_heads=8,
                 num_layers=6,
                 dropout=0.1,
                 activation="relu",
                 return_intermediate=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(embed_dims, num_heads, feed_dims, dropout, activation)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dims)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class PlainTransformerDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dims: int,
        num_queries: int,
        num_headss: int,
        dropout: float,
        feed_dims: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        super().__init__()

        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dims // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        transformer = Transformer(
            embed_dims=hidden_dims,
            dropout=dropout,
            num_heads=num_headss,
            feed_dims=feed_dims,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dims = transformer.embed_dims

        self.query_embed = nn.Embedding(num_queries, hidden_dims)

        if in_channels != hidden_dims or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dims, kernel_size=1)
            nn.init.kaiming_uniform_(self.input_proj.weight, a=1)
            if self.input_proj[-1].bias is not None:
                nn.init.constant_(self.input_proj[-1].bias, 0)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dims, num_classes + 1)
        self.mask_embed = MLP(hidden_dims, hidden_dims, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        pos = self.pe_layer(x, mask)

        src = x
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(outputs_class if self.mask_classification else None,
                                                    outputs_seg_masks)
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
