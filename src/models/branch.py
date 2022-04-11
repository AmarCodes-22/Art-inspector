import torch
from detectron2.config.config import CfgNode
from torch import nn

from .. import ARTNET_CONFIG_FPATH
from ..utils import load_config


class LinearBranch(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_dims: int = 256
    ) -> None:
        # todo: write docstring
        super(LinearBranch, self).__init__()
        self.input_size = in_features
        self.linear1 = nn.Linear(in_features, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# todo: find out how to share weights when the input feature map is of different sizes?
# one idea is to avgpool all feature maps to a common size
class FPNBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_features: int,
        hidden_channels: int = 128,
    ) -> None:
        # todo: write docstring
        super(FPNBranch, self).__init__()
        self.resize_layer = nn.AdaptiveAvgPool2d((7, 7))
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3)
        self.linear = nn.Linear(out_channels, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resize_layer(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.linear(x))
        return x


def build_branch(cfg: CfgNode, branch_name: str):
    branch_config = dict(cfg.MODELS.BRANCH.BRANCHES)[branch_name.upper()]

    # output channels double every stage
    stem_out_size = cfg.MODELS.RESNET.RES2_OUT_CHANNELS * (2 ** 3)

    branch_num_classes = branch_config.NUM_CLASSES

    assert cfg.MODELS.BRANCH.BRANCH_TYPE in {"fpn", "linear"}
    if cfg.MODELS.BRANCH.BRANCH_TYPE == "fpn":
        return FPNBranch(
            in_channels=128, out_channels=128, out_features=branch_num_classes
        )
    else:
        return LinearBranch(in_features=stem_out_size, out_features=branch_num_classes)


if __name__ == "__main__":
    artnet_config = load_config(ARTNET_CONFIG_FPATH)

    branch = build_branch(artnet_config, branch_name="artists")
    print(type(branch))  # show which branch was dispatched

    if artnet_config.MODELS.BRANCH.BRANCH_TYPE == "fpn":
        dummy_in = torch.zeros((1, 128, 7, 7))
    else:
        dummy_in = torch.zeros((1, artnet_config.MODELS.RESNET.RES2_OUT_CHANNELS * 8))

    dummy_out = branch(dummy_in)
    print(dummy_out.shape)
