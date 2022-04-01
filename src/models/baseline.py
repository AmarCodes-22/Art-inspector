from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import Identity
from torchinfo import summary
from torchvision.models.resnet import BasicBlock, ResNet


class Stem(nn.Module):
    def __init__(self, layers: List[int] = [2, 2, 2, 2]) -> None:
        """Initialize a ResNet stem for baseline composed of BasicBlocks

        Parameters
        ----------
        layers : List[int], optional
            Each item in layers indicates a stage and the value for that stage indicates
            the number of BasicBlocks in that stage, by default [2, 2, 2, 2] to create a
            ResNet18 architecture
        """
        super(Stem, self).__init__()
        self.model = ResNet(block=BasicBlock, layers=layers)
        self.model.fc = Identity()

    def forward(self, x):
        return self.model(x)

    def print_summary(self, input_size: Tuple[int, int, int, int]):
        """Print summary using torchinfo

        Parameters
        ----------
        input_size : Tuple[int, int, int, int]
            Input image shape: (Batch_size, num_channels, height, width)
        """
        # summary(self.model, input_size=(16, 3, 28, 28))
        summary(self.model, input_size=input_size)

    def get_output_size(self, input_size: Tuple[int, int, int, int]):
        """Get output size of the stem for the input size provided

        Parameters
        ----------
        input_size : Tuple[int, int, int, int]
            Size of input image: (batch_size, num_channels, height, width)

        Returns
        -------
        int
            Number of output features after passing image of 'input_size' through stem
        """
        dummy_input = torch.zeros(input_size)
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

    def forward(self, x, branch_name: Optional[str] = None):
        stem_out = self.stem(x)

        if branch_name is None:
            outputs_dict = dict()
            output_tensors = []

            # below would probably now work for branches with different num_classes
            # output = torch.empty(size=(len(self.branch_configs), ))

            for branch_name, (in_features, out_features) in self.branch_configs.items():
                branch_name += "_branch"
                branch = getattr(self, branch_name)

                outputs_dict[branch] = branch(stem_out)
                output_tensors.append(outputs_dict[branch])

            # todo: this should probably return a torch.Tensor
            return torch.cat(output_tensors, dim=1)
        else:
            branch_name += "_branch"
            branch = getattr(self, branch_name)
            output_tensor = branch(stem_out)
            return output_tensor

    def initialize_branches(self, branch_configs: Dict[str, Tuple[int, int]]):
        """Initialize branches specific to each branch in the dataset

        Parameters
        ----------
        branch_configs : Dict[str, Tuple[int, int]]
            dict mapping from 'branch_name' to (in_features, out_features)
            out_features is equal to number of classes for that branch
        """
        self.num_branches = len(branch_configs.keys())
        self.branch_configs = branch_configs
        for branch_name, (in_features, out_features) in self.branch_configs.items():
            branch_name += "_branch"
            setattr(self, branch_name, self._make_branch(in_features, out_features))

    def _make_stem(self, layers):
        stem = Stem(layers=layers)
        return stem

    def _make_branch(self, in_features: int, out_features: int, hidden_dims: int = 256):
        branch = Branch(in_features, out_features, hidden_dims)
        return branch

    def print_model_summary(self, input_size: Tuple[int, int, int, int]):
        print("Printing model summary")
        print(summary(self, input_size=input_size))


if __name__ == "__main__":

    # makes a resnet18 stem and 3 MLP task specific heads
    resnet18_layers = [2, 2, 2, 2]
    baseline = Baseline(stem_layers=resnet18_layers)

    stem_out_size = baseline.stem.get_output_size()

    # this information should come from your ImageFolder dataset
    branches = {
        "artist": (stem_out_size, 20),
        "genre": (stem_out_size, 10),
        "style": (stem_out_size, 20),
    }

    baseline.initialize_branches(branches=branches)
    baseline.print_model_summary()
