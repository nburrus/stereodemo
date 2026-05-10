# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Sourced from https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .third_party import DPTHead


def _reg_dense_depth(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    return xyz * (torch.exp(d) - 1)


def _reg_dense_conf(x, vmin=1):
    return vmin + x.exp()


def _postprocess(out):
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,4
    depth = _reg_dense_depth(fmap[:, :, :, 0:3])
    conf = _reg_dense_conf(fmap[:, :, :, 3])
    return depth, conf


class LinearPts3d(nn.Module):
    def __init__(self, width=512, height=512, patch_size=16, dec_embed_dim=768, has_conf=True):
        super().__init__()
        self.patch_size = patch_size
        self.has_conf = has_conf
        self.num_h = height // patch_size
        self.num_w = width // patch_size
        self.proj = nn.Linear(dec_embed_dim, (3 + has_conf) * patch_size ** 2)

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):
        B, S, D = tokens_12.shape
        feat = self.proj(tokens_12)
        feat = feat.transpose(-1, -2).view(B, -1, self.num_h, self.num_w)
        return F.pixel_shuffle(feat, self.patch_size)


class Dust3rHead(nn.Module):
    def __init__(self, ckpt_dict, width=512, height=512, device=torch.device('cpu')):
        super().__init__()
        is_dpt = any('dpt' in k for k in ckpt_dict['model'].keys())
        self.downstream_head1 = DPTHead(width, height) if is_dpt else LinearPts3d(width, height)
        self.downstream_head2 = DPTHead(width, height) if is_dpt else LinearPts3d(width, height)
        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12):
        out1 = self.downstream_head1(d1_0, d1_6, d1_9, d1_12)
        out2 = self.downstream_head2(d2_0, d2_6, d2_9, d2_12)
        pts3d1, conf1 = _postprocess(out1)
        pts3d2, conf2 = _postprocess(out2)
        return pts3d1, conf1, pts3d2, conf2

    def _load_checkpoint(self, ckpt_dict):
        state = {k.replace('.dpt', ''): v for k, v in ckpt_dict['model'].items() if 'head' in k}
        self.load_state_dict(state, strict=True)
