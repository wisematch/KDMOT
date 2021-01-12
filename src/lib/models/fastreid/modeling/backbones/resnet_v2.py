# encoding: utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

# based on:
# https://github.com/google-research/big_transfer/blob/master/bit_pytorch/models.py

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
from ...utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import logging
import torch.nn.functional as F
from .build import BACKBONE_REGISTRY


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, last_stride, block_units, width_factor):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64 * wf, cout=256 * wf, cmid=64 * wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256 * wf, cout=256 * wf, cmid=64 * wf)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256 * wf, cout=512 * wf, cmid=128 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512 * wf, cout=512 * wf, cmid=128 * wf)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512 * wf, cout=1024 * wf, cmid=256 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024 * wf, cout=1024 * wf, cmid=256 * wf)) for i in
                 range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024 * wf, cout=2048 * wf, cmid=512 * wf, stride=last_stride))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048 * wf, cout=2048 * wf, cmid=512 * wf)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))
        # pylint: enable=line-too-long

    def forward(self, x):
        x = self.body(self.root(x))
        return x
        # assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        # return x[..., 0, 0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
            # self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            # self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            # if self.zero_head:
            #     nn.init.zeros_(self.head.conv.weight)
            #     nn.init.zeros_(self.head.conv.bias)
            # else:
            #     self.head.conv.weight.copy_(
            #         tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
            #     self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2(1, [3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])


@BACKBONE_REGISTRY.register()
def build_resnetv2_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm = cfg.MODEL.BACKBONE.NORM
    num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
    with_ibn = cfg.MODEL.BACKBONE.WITH_IBN
    with_se = cfg.MODEL.BACKBONE.WITH_SE
    with_nl = cfg.MODEL.BACKBONE.WITH_NL
    depth = cfg.MODEL.BACKBONE.DEPTH

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], }[depth]
    nl_layers_per_stage = {50: [0, 2, 3, 0], 101: [0, 2, 9, 0]}[depth]
    # model = ResNetV2(last_stride, bn_norm, num_splits, with_ibn, with_se, with_nl, Bottleneck,
    #                num_blocks_per_stage, nl_layers_per_stage)
    model = ResNetV2(last_stride, num_blocks_per_stage, 1)
    if pretrain:
        import numpy as np
        state_dict = np.load(pretrain_path)
        model.load_from(state_dict)

    return model
