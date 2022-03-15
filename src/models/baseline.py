from pprint import pprint
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchinfo import summary
from torch.nn import Identity
from torchvision.models.resnet import ResNet, BasicBlock


class Stem(nn.Module):
    def __init__(self, layers: List[int] = [2, 2, 2, 2]) -> None:
        """Initialize a ResNet stem for baseline composed of BasicBlocks

        Parameters
        ----------
        layers : List[int], optional
            Each item in layers indicated a stage and the value for that stage indicates
            the number of BasicBlocks in that stage, by default [2, 2, 2, 2] to create a
            ResNet18 architecture
        """
        super(Stem, self).__init__()
        self.model = ResNet(block=BasicBlock, layers=layers)
        self.model.fc = Identity()

    def forward(self, x):
        return self.model(x)

    def print_summary(self):
        summary(self.model, input_size=(16, 3, 224, 224))

    def get_output_size(self):
        dummy_input = torch.zeros(1, 3, 224, 224)
        return self.forward(dummy_input).data.size()[1]


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
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, out_features),
        )

    def forward(self, x):
        return self.model(x)

    def print_summary(self):
        summary(self.model, input_size=(1, self.input_size))

    def get_output_size(self):
        dummy_input = torch.zeros(1, self.input_size)
        return self.forward(dummy_input).data.size()


class Baseline(nn.Module):
    def __init__(
        self,
        stem_layers: List[int] = [2, 2, 2, 2],
    ) -> None:
        """Initialize baseline

        Parameters
        ----------
        stem_layers: List[int]
            A list of stages in a ResNet network where value for each stage is the
            number of BasicBlocks for that stage
        """
        super(Baseline, self).__init__()

        self.stem = self._make_stem(layers=stem_layers)

    def forward(self, x):
        stem_out = self.stem(x)

        artist_out = self.artist_branch(stem_out)
        genre_out = self.genre_branch(stem_out)
        style_out = self.style_branch(stem_out)

        outputs_dict = dict()
        outputs_dict["artist"] = artist_out
        outputs_dict["genre"] = genre_out
        outputs_dict["style"] = style_out

        return outputs_dict

    def initialize_branches(self, branches: Dict[str, Tuple[int, int]]):
        self.num_branches = len(branches)
        for branch_name, (in_features, out_features) in branches.items():
            branch_name += "_branch"
            setattr(self, branch_name, self._make_branch(in_features, out_features))

    def _make_stem(self, layers):
        stem = Stem(layers=layers)
        return stem

    def _make_branch(self, in_features: int, out_features: int, hidden_dims: int = 256):
        branch = Branch(in_features, out_features, hidden_dims)
        return branch

    def print_model_summary(self):
        print('Printing model summary')
        summary(self, input_size=(1, 3, 224, 224))


if __name__ == "__main__":

    # makes a resnet18 stem and 3 MLP task specific heads
    resnet18_layers = [2, 2, 2, 2]
    baseline = Baseline(stem_layers=resnet18_layers)

    stem_out_size = baseline.stem.get_output_size()

    branches = {
        "artist": (stem_out_size, 20),
        "genre": (stem_out_size, 10),
        "style": (stem_out_size, 20)
    }

    baseline.initialize_branches(branches=branches)
    baseline.print_model_summary()

    pprint(baseline(torch.zeros(1, 3, 224, 224)))
