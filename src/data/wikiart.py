import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .. import ARTNET_CONFIG_FPATH
from ..utils import load_config


class WikiartBranch(Dataset):
    def __init__(
        self,
        branch_name: str,
        split: str,
        dir_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize dataset for branch dataset.

        Parameters
        ----------
        branch_name: str
            Name of the branch of dataset, must be one of 'artists', 'genres', 'styles'
        split : str
            One of 'train' or 'test'
        dir_path : Optional[Union[str, Path, None]], optional
            Path to directory containing images for the branch,
            by default project_root/data/wikiart_datasets/{branch_name}/images/{split}
        """
        super(WikiartBranch, self).__init__()

        self._validate(argument="branch_name", argument_value=branch_name)
        self._validate(argument="split", argument_value=split)
        self._validate(argument="dir_path", argument_value=dir_path)

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
        self.dataset = ImageFolder(
            os.path.join(self.images_dir, split), transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def _validate(self, argument: str, argument_value: Optional[str] = None):
        """Validate arguments provided to initialize WikiartBranch.

        Parameters
        ----------
        argument : str
            Argument to validate
        argument_value: str, optional
            Value to validate the argument at, by default None

        Note: branch_name must be validated before dir_path
        """
        if argument == "branch_name":
            if argument_value not in {"artists", "genres", "styles"}:
                raise ValueError(
                    "Wrong branch name provided, must be one of 'artists', 'genres' or"
                    " 'styles'."
                )
            else:
                self.name = argument_value

        elif argument == "split":
            if argument_value not in {"train", "test"}:
                raise ValueError(
                    "Wrong split provided, must be one of 'train' or 'test'."
                )
            else:
                self.split = argument_value

        elif argument == "dir_path":
            if argument_value is None:
                self.images_dir = os.path.join(
                    Path(__file__).parents[2],
                    "data",
                    "wikiart_datasets",
                    self.name,
                    "images",
                )
            else:
                self.images_dir = argument_value


class Wikiart:
    def __init__(self, split: str) -> None:

        self.branch_names = ("artists", "styles", "genres")
        self.branch_batch_sizes = dict(load_config(ARTNET_CONFIG_FPATH).DATA.BATCH_SIZE)
        # print(self.branch_batch_sizes)
        # self.branch_batch_sizes = {"artists": 16, "styles": 32, "genres": 32}

        for branch_name in self.branch_names:
            branch_dataset = WikiartBranch(branch_name, split)
            setattr(self, branch_name + "_dataset", branch_dataset)
            print(
                f"Loaded {split} dataset for branch {branch_name} with"
                f" {len(getattr(self, branch_name + '_dataset').dataset)} images."
            )

    def load_dataloaders(self):
        dataloaders = {}
        for branch_name in self.branch_names:
            branch_dataloader = DataLoader(
                getattr(self, branch_name + "_dataset").dataset,
                batch_size=self.branch_batch_sizes[branch_name.upper()],
                shuffle=True,
            )
            dataloaders[branch_name] = branch_dataloader

        return dataloaders

    def __len__(self):
        return (
            len(self.artists_dataset.dataset)
            + len(self.genres_dataset.dataset)
            + len(self.styles_dataset.dataset)
        )


if __name__ == "__main__":
    wikiart = Wikiart(split="train")
    wikiart_dataloaders = wikiart.load_dataloaders()

    # pprint(wikiart_dataloaders)

    images, labels = next(iter(wikiart_dataloaders["artists"]))
    print("Artists branch -", images.shape, labels.shape)

    images, labels = next(iter(wikiart_dataloaders["genres"]))
    print("Genres branch -", images.shape, labels.shape)

    images, labels = next(iter(wikiart_dataloaders["styles"]))
    print("Styles branch -", images.shape, labels.shape)
