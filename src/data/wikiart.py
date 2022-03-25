import os

from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


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

        self.transform = transforms.Compose([transforms.ToTensor()])
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


class Wikiart(Dataset):
    def __init__(self, split: str) -> None:
        super(Wikiart, self).__init__()

        self.branch_names = ("artists", "styles", "genres")

        for branch_name in self.branch_names:
            branch_dataset = WikiartBranch(branch_name, split)
            setattr(self, branch_name + "_branch", branch_dataset)
            print(
                f"Loaded {split} dataset for branch {branch_name} with"
                f" {len(getattr(self, branch_name + '_branch').dataset)} images."
            )

    def __len__(self):  # num of images total
        return (
            len(self.artists_branch.dataset)
            + len(self.genres_branch.dataset)
            + len(self.styles_branch.dataset)
        )

    def __getitem__(self, index):  # loads sequentially from WikiartBranch
        pass

    def __repr__(self) -> str:  # shows details about all branches
        pass


if __name__ == "__main__":
    wikiart = Wikiart(split="train")
