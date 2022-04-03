import torch
from torch import nn

from ..utils.util import load_config
from . import ARTNET_CONFIG_FPATH
from .branch import build_branch
from .resnet import build_resnet_backbone


class ArtNet(nn.Module):
    def __init__(self, config_fpath: str) -> None:
        super(ArtNet, self).__init__()
        self.config = load_config(config_fpath)
        self.resnet_stem = build_resnet_backbone(self.config)
        self.branch_names = list(
            map(str.lower, list(dict(self.config.MODEL.BRANCH).keys()))
        )

        for branch_name in self.branch_names:
            branch = build_branch(cfg=self.config, branch_name=branch_name)
            branch_name = branch_name + "_branch"
            setattr(self, branch_name, branch)

    def forward(self, x, branch_name):
        outputs = {}

        x = self.resnet_stem(x)["linear"]

        branch_out = getattr(self, branch_name + "_branch")(x)
        return branch_out

        # for branch_name in self.branch_names:
        #     branch_out = getattr(self, branch_name + "_branch")(x)
        #     outputs[branch_name] = branch_out

        return outputs


if __name__ == "__main__":
    artnet = ArtNet(config_fpath=ARTNET_CONFIG_FPATH)

    dummy_in = torch.zeros((1, 3, 224, 224))
    dummy_out = artnet(dummy_in)

    for k, v in dummy_out.items():
        print(k, type(v), v.shape)
