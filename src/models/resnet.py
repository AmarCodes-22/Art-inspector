# import os
# from pathlib import Path
from typing import List

import numpy as np
import torch
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import BasicBlock, BasicStem
from torch import nn

from .. import ARTNET_CONFIG_FPATH
from ..utils import load_config


class Resnet(nn.Module):
    """
    Implements a resnet backbone of the network. (only R18/R34 for now)
    """

    def __init__(
        self,
        stem: BasicStem,
        stages: List[List[BasicBlock]],
        out_features: List[str]
        # num_classes: Optional[int] = None,
    ) -> None:
        super(Resnet, self).__init__()
        assert len(stages) > 0, "Length of stages received can't be 0"

        self.stem = stem
        # self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        stages = self._get_relevant_stages(stages, out_features)

        for i, blocks in enumerate(stages):
            assert (
                len(blocks) > 0
            ), f"Stage {i+1} should contain atleast 1 block, got {len(blocks)} blocks."

            for block in blocks:
                assert isinstance(block, (BasicBlock)), (
                    "Block should be an instance of"
                    " detectron2.modeling.backbone.resnet.BasicBlock"
                )

            stage_name = "res" + str(i + 2)  # res2's output corresponds to first stage
            stage = nn.Sequential(*blocks)

            self._add_stage(stage_name, stage)

            self._out_feature_strides[stage_name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            curr_channels = blocks[-1].out_channels
            self._out_feature_channels[stage_name] = curr_channels
        self.stage_names = tuple(self.stage_names)

        if "linear" in out_features:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            name = "linear"
            self._out_feature_channels[name] = curr_channels

        # todo: check this, does this get 'name' variable which doesn't exist
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        self._check_validity_of_output_features()

    def forward(self, x):
        assert (
            x.dim() == 4
        ), f"Resnet takes an input of size (N, C, H, W), got {x.shape} instead."

        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        for stage_name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if stage_name in self._out_features:
                outputs[stage_name] = x

        if "linear" in self._out_features:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            outputs["linear"] = x

        return outputs

    def output_shape(self):
        output_shapes = dict()
        for name in self._out_features:
            if name == "linear":
                output_shapes[name] = ShapeSpec(
                    channels=self._out_feature_channels[name]
                )
                continue
            output_shapes[name] = ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
        return output_shapes

    @staticmethod
    def make_stage(
        block_class: BasicBlock,
        num_blocks: int,
        *,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        """
        Create a list of blocks that form one Resnet stage
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument {k} of make_stage should have same length as"
                        f" num_blocks={num_blocks}"
                    )
                    newk = k[: -len("_per_block")]
                    assert (
                        newk not in kwargs
                    ), f"Cannot call make_stage with both {k} and {newk}"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(block_class(in_channels, out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks

    def _check_validity_of_output_features(self):
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            if out_feature == "linear":
                continue
            assert out_feature in children, "Available children: {}, got {}".format(
                ", ".join(children), out_feature
            )

    def _get_relevant_stages(self, stages, out_features):
        if out_features is not None:
            num_stages_to_keep = max(
                [
                    {"res2": 1, "res3": 2, "res4": 3, "res5": 4, "linear": 4}.get(f, 0)
                    for f in out_features
                ]
            )
        stages = stages[:num_stages_to_keep]
        return stages

    def _add_stage(self, stage_name, stage):
        self.add_module(stage_name, stage)
        self.stage_names.append(stage_name)
        self.stages.append(stage)


def build_resnet_backbone(cfg):
    """Create a Resnet instance from config"""
    resnet_stem = BasicStem()

    depth = cfg.MODELS.RESNET.DEPTH
    assert depth in {18, 34}, "Only support depth 18 and 34 for now"
    out_features = cfg.MODELS.RESNET.OUT_FEATURES

    in_channels = cfg.MODELS.RESNET.STEM_OUT_CHANNELS
    out_channels = cfg.MODELS.RESNET.RES2_OUT_CHANNELS

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        # 50: [3, 4, 6, 3],
        # 101: [3, 4, 23, 3],
        # 152: [3, 8, 36, 3],
    }[depth]

    assert (
        out_channels == 64
    ), "Must set models.resnet.res2_out_channels = 64 for R18/R34"

    stages = []
    for idx, stage_idx in enumerate(range(2, 6)):
        first_stride = 1 if idx == 0 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
        }

        stage_kargs["block_class"] = BasicBlock
        blocks = Resnet.make_stage(**stage_kargs)

        in_channels = out_channels
        out_channels *= 2

        stages.append(blocks)

    CNN = Resnet(stem=resnet_stem, stages=stages, out_features=out_features)
    return CNN


# if __name__ == "__main__":
#     artnet_config = load_config(filepath=ARTNET_CONFIG_FPATH)
#     resnet_backbone = build_resnet_backbone(artnet_config)

#     dummy_in = torch.zeros((1, 3, 224, 224))
#     dummy_out = resnet_backbone(dummy_in)
#     for k, v in dummy_out.items():
#         print(k, type(v), v.shape)
