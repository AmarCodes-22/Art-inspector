"""
Implements a Resnet18 stem with 3 MLP task specific heads
"""
import torch
from torch import nn
from torchviz import make_dot
from torchinfo import summary
from torch.nn import Identity
from torchvision.models.resnet import ResNet, BasicBlock


class Stem(nn.Module):
    def __init__(self) -> None:
        super(Stem, self).__init__()
        self.model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
        self.model.fc = Identity()

    def forward(self, x):
        return self.model(x)

    def print_summary(self):
        summary(self.model, input_size=(16, 3, 224, 224))

    def get_output_size(self):
        dummy_input = torch.zeros(1, 3, 224, 224)
        return self.forward(dummy_input).data.size()


class Branch(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features) -> None:
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
    def __init__(self) -> None:
        super(Baseline, self).__init__()
        self.stem = self._make_stem()
        self.artist_branch = self._make_branch(512, 256, 20)
        self.genre_branch = self._make_branch(512, 256, 10)
        self.style_branch = self._make_branch(512, 256, 20)

    def forward(self, x):
        stem_out = self.stem(x)

        artist_out = self.artist_branch(stem_out)
        genre_out = self.genre_branch(stem_out)
        style_out = self.style_branch(stem_out)

        outputs_dict = dict()
        outputs_dict['artist'] = artist_out
        outputs_dict['genre'] = genre_out
        outputs_dict['style'] = style_out

        return outputs_dict

    def get_output_size(self):
        dummy_input = torch.zeros(1, 3, 224, 224)
        return self.forward(dummy_input).data.size()

    def _make_stem(self):
        stem = Stem()
        return stem

    def _make_branch(self, in_features, hidden_dims, out_features):
        branch = Branch(in_features, hidden_dims, out_features)
        return branch


if __name__ == "__main__":
    baseline = Baseline()

    # summary(baseline, input_size=(1, 3, 224, 224))
    # print(baseline.get_output_size())

    # baseline.stem.print_summary()
    # print(baseline.stem.get_output_size())

    # baseline.artist_branch.print_summary()
    # print(baseline.artist_branch.get_output_size())

    # baseline.genre_branch.print_summary()
    # print(baseline.genre_branch.get_output_size())

    # baseline.style_branch.print_summary()
    # print(baseline.style_branch.get_output_size())

    # graphviz plot for baseline
    # x = torch.zeros(1, 3, 224, 224)
    # y = baseline(x)
    # plot = make_dot(y.mean(), params=dict(baseline.named_parameters()))
    # plot.render(filename='baseline.dot')
