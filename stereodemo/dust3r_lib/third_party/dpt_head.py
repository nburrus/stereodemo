# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Sourced from https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal
# References: https://github.com/isl-org/DPT
#             https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union, Tuple


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer_rn = nn.ModuleList([scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn])
    return scratch


class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, width_ratio=1):
        super().__init__()
        self.width_ratio = width_ratio
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if self.width_ratio != 1:
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output = F.interpolate(output, size=(2 * output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False,
                                     align_corners=True, width_ratio=width_ratio)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class DPTHead(nn.Module):
    def __init__(self, width=512, height=512, num_channels: int = 4, stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16, layer_dims: Tuple[int] = (96, 192, 384, 768),
                 feature_dim: int = 256, last_dim: int = 128, use_bn: bool = False,
                 dim_tokens_enc: Tuple[int] = (1024, 768, 768, 768), output_width_ratio=1, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)
        self.num_w = width // (self.stride_level * self.P_W)
        self.num_h = height // (self.stride_level * self.P_H)
        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)
        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
        )
        self.act_postprocess = self._init_act_postprocess()

    def _init_act_postprocess(self):
        act_postprocess = nn.ModuleList()
        act_postprocess.append(nn.Sequential(
            nn.Conv2d(self.dim_tokens_enc[0], self.layer_dims[0], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(self.layer_dims[0], self.layer_dims[0], kernel_size=4, stride=4, padding=0, bias=True),
        ))
        act_postprocess.append(nn.Sequential(
            nn.Conv2d(self.dim_tokens_enc[1], self.layer_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(self.layer_dims[1], self.layer_dims[1], kernel_size=2, stride=2, padding=0, bias=True),
        ))
        act_postprocess.append(nn.Sequential(
            nn.Conv2d(self.dim_tokens_enc[2], self.layer_dims[2], kernel_size=1, stride=1, padding=0),
        ))
        act_postprocess.append(nn.Sequential(
            nn.Conv2d(self.dim_tokens_enc[3], self.layer_dims[3], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.layer_dims[3], self.layer_dims[3], kernel_size=3, stride=2, padding=1),
        ))
        return act_postprocess

    def forward(self, tokens_0, tokens_6, tokens_9, tokens_12):
        layers = [tokens_0, tokens_6, tokens_9, tokens_12]
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=self.num_h, nw=self.num_w) for l in layers]
        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        return self.head(path_1)
