import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from torch import nn

from ..utils.util import load_config
from . import ARTNET_CONFIG_FPATH
from .resnet import build_resnet_backbone


class FPN(nn.Module):
    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        # top_block=None,
        # fuse_type="sum",
    ) -> None:
        super(FPN, self).__init__()
        assert isinstance(bottom_up, nn.Module)
        # assert in_features, in_features

        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        self._assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias
            )
            output_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias
            )

            stage_num = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage_num), lateral_conv)
            self.add_module("fpn_output{}".format(stage_num), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        # self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }

        # if self.top_block is not None:
        #     for s in range(stage, stage + self.top_block_num_levels):
        #         self._out_feature_strides['p{}'.format(s+1)] = 2 ** (s+1)

        self._out_feature_names = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_feature_names}

        # assert fuse_type in {"avg", "sum"}
        # self._fuse_type = fuse_type

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []

        # calculate features using the last output of bottom_up
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # we skip the first one because we've already used it to compute above
            if idx > 0:
                features_name = self.in_features[-idx - 1]
                features = bottom_up_features[features_name]

                # scale up prev_features to match resolution of lateral_convs
                top_down_features = F.interpolate(
                    prev_features, scale_factor=2.0, mode="nearest"
                )
                lateral_features = lateral_conv(features)
                prev_features = top_down_features + lateral_features

                results.insert(0, output_conv(prev_features))

        assert len(self._out_feature_names) == len(results)
        return {f: res for f, res in zip(self._out_feature_names, results)}

    def _assert_strides_are_log2_contiguous(self, strides):
        for i, stride in enumerate(strides[1:], 1):
            assert (
                stride == 2 * strides[i - 1]
            ), "Strides {} {} are not log2 contiguous".format(stride, strides[i - 1])


def build_resnet_fpn_backbone(cfg):
    bottom_up = build_resnet_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up, in_features=in_features, out_channels=out_channels
    )
    return backbone


if __name__ == "__main__":
    artnet_config = load_config(filepath=ARTNET_CONFIG_FPATH)
    fpn_backbone = build_resnet_fpn_backbone(artnet_config)
    print(type(fpn_backbone))

    dummy_in = torch.zeros((1, 3, 224, 224))
    dummy_out = fpn_backbone(dummy_in)
    for k, v in dummy_out.items():
        print(k, type(v), v.shape)
