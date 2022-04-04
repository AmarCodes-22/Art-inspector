import torch
from torch import nn

from src.models.fpn import build_resnet_fpn_backbone

from ..utils import load_config
from . import ARTNET_CONFIG_FPATH
from .branch import build_branch
from .resnet import build_resnet_backbone


class ArtNet(nn.Module):
    def __init__(self, config_fpath: str) -> None:
        super(ArtNet, self).__init__()

        self.config = load_config(config_fpath)
        self.stem = self._initialize_stem()
        self.branch_names = list(
            map(str.lower, list(dict(self.config.MODEL.BRANCH.BRANCHES).keys()))
        )

        for branch_name in self.branch_names:
            branch = build_branch(cfg=self.config, branch_name=branch_name)
            branch_name = branch_name + "_branch"
            setattr(self, branch_name, branch)

    def forward(self, x, branch_name):
        # todo: implement this for multiple features maps
        x = self.stem(x)[self.stem._out_feature_names[0]]

        branch_out = getattr(self, branch_name + "_branch")(x)
        return branch_out

    def _initialize_stem(self):
        if self.config.MODEL.BRANCH.BRANCH_TYPE == "linear":
            stem = build_resnet_backbone(self.config)
        else:
            stem = build_resnet_fpn_backbone(self.config)
        return stem


if __name__ == "__main__":
    artnet = ArtNet(config_fpath=ARTNET_CONFIG_FPATH)

    dummy_in = torch.zeros((1, 3, 224, 224))
    dummy_out = artnet(dummy_in, "artists")

    print(dummy_out.shape)
