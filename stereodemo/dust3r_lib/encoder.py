# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Sourced from https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal

from functools import partial

import torch
import torch.nn as nn

from .third_party import RoPE2D
from .blocks import DropPath, Mlp, Attention


class Block(nn.Module):
    def __init__(self, dim, num_heads, rope, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(512, 512), patch_size=(16, 16), in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return self.norm(x)


class Dust3rEncoder(nn.Module):
    def __init__(self, ckpt_dict, batch=2, width=512, height=512, patch_size=16,
                 enc_embed_dim=1024, enc_num_heads=16, enc_depth=24, mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), device=torch.device('cpu')):
        super().__init__()
        self.patch_embed = PatchEmbed((height, width), (patch_size, patch_size), 3, enc_embed_dim)
        self.rope = RoPE2D(batch, width, height, patch_size, base=100.0, device=device)
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, self.rope, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(enc_depth)
        ])
        self.enc_norm = norm_layer(enc_embed_dim)
        self._load_checkpoint(ckpt_dict)
        self.to(device)

    @torch.inference_mode()
    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_norm(x)

    def _load_checkpoint(self, ckpt_dict):
        state = {k: v for k, v in ckpt_dict['model'].items()
                 if k.startswith(('patch_embed', 'enc_blocks', 'enc_norm'))}
        self.load_state_dict(state, strict=True)
