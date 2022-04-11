import torch
from torch import nn

from src.models.fpn import build_resnet_fpn_backbone

from .. import ARTNET_CONFIG_FPATH
from ..utils import load_config
from .branch import build_branch
from .resnet import build_resnet_backbone


class ArtNet(nn.Module):
    def __init__(self, config_fpath: str) -> None:
        super(ArtNet, self).__init__()

        self.config = load_config(config_fpath)
        self.stem = self._initialize_stem()
        self.branch_names = list(
            map(str.lower, list(dict(self.config.MODELS.BRANCH.BRANCHES).keys()))
        )
        self.branch_type = self.config.MODELS.BRANCH.BRANCH_TYPE

        for branch_name in self.branch_names:
            branch = build_branch(cfg=self.config, branch_name=branch_name)
            branch_name = branch_name + "_branch"
            setattr(self, branch_name, branch)

    def forward(self, x, branch_name):
        if self.branch_type == "fpn":
            x = self.stem(x)  # returns a dict of fpn features

            output = dict()
            for feature_name, features in x.items():
                output[feature_name] = getattr(self, branch_name + "_branch")(features)

            return torch.cat(list(output.values()), 1)
        else:
            x = self.stem(x)["linear"]
            output = getattr(self, branch_name + "_branch")(x)
            return output

    def _initialize_stem(self):
        if self.config.MODELS.BRANCH.BRANCH_TYPE == "linear":
            stem = build_resnet_backbone(self.config)
        else:
            stem = build_resnet_fpn_backbone(self.config)
        return stem


if __name__ == "__main__":
    artnet = ArtNet(config_fpath=ARTNET_CONFIG_FPATH)

    dummy_in = torch.zeros((1, 3, 224, 224))
    dummy_out = artnet(dummy_in, "artists")
    if isinstance(dummy_out, torch.Tensor):
        print(dummy_out.shape)
    elif isinstance(dummy_out, dict):
        # print(torch.cat(list(dummy_out.values()), 1).shape)
        for k, v in dummy_out.items():
            print(k, v.shape)
