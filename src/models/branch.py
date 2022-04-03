from detectron2.config.config import CfgNode
from torch import nn
import torch

from ..utils.util import load_config
from . import ARTNET_CONFIG_FPATH


class Branch(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_dims: int = 256
    ) -> None:
        """Initialize a MLP for the task specific head

        Parameters
        ----------
        in_features : int
            Input features to the MLP, should be same as out_features from Stem
        out_features : int
            Out features to the MLP, should be equal to the number of classes for that
            task
        hidden_dims : int, optional
            Number of units in the hidden layer of MLP, by default 256
        """
        super(Branch, self).__init__()
        self.input_size = in_features
        self.linear1 = nn.Linear(in_features, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def build_branch(cfg: CfgNode, branch_name: str):
    branch_config = dict(cfg.MODEL.BRANCH)[branch_name.upper()]

    # output channels double every stage
    stem_out_size = cfg.MODEL.RESNET.RES2_OUT_CHANNELS * (2 ** 3)

    branch_num_classes = branch_config.NUM_CLASSES
    return Branch(in_features=stem_out_size, out_features=branch_num_classes)


if __name__ == '__main__':
    artnet_config = load_config(ARTNET_CONFIG_FPATH)
    # print(artnet_config)

    branch = build_branch(artnet_config, branch_name='artists')

    dummy_in = torch.zeros((1, artnet_config.MODEL.RESNET.RES2_OUT_CHANNELS * 8))
    dummy_out = branch(dummy_in)
    print(dummy_out.shape)
