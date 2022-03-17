from pathlib import Path
from os import listdir
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Mapping
from os.path import join, isdir

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class Cifar100:
    def __init__(
        self, dir_path: Optional[str] = None, branch_names: Optional[List[str]] = None
    ) -> None:
        """Initialize Cifar100 dataset

        Parameters
        ----------
        dir_path : Optional[str]
            Path to the directory where this dataset is located
        branch_names : Optional[List[str]]
            List of branch names to load the dataset for
        """
        if dir_path is not None:
            self.dir_path = dir_path
        else:
            self.dir_path = join(Path(__file__).parents[2], "data", "cifar100")

        if branch_names is None:
            self.branch_names = listdir(join(self.dir_path, "train"))
        else:
            self.valid_branch_names = set(listdir(join(self.dir_path, "train")))
            for branch_name in branch_names:
                if branch_name not in self.valid_branch_names:
                    print(
                        "{} is not a valid branch name. Should be one of {}".format(
                            branch_name, self.valid_branch_names
                        )
                    )
                break
            self.branch_names = branch_names

        self.branch_datasets = None
        self.transform = transforms.Compose([transforms.ToTensor()])

    def load_branch_datasets(self) -> Mapping[str, Mapping[str, ImageFolder]]:
        """Load datasets for all branches inside Cifar100

        Returns
        -------
        dict: Mapping[str, Mapping[str, ImageFolder]]
            Dictionary mapping from each branch's train and test to torchvision.datasets.ImageFolder instance
        """
        datasets = defaultdict(dict)

        for i, branch in enumerate(self.branch_names):
            # print(f"{i}. Creating dataset for {branch}")

            branch_train_dir = join(self.dir_path, "train", branch)
            branch_test_dir = join(self.dir_path, "test", branch)

            if isdir(branch_train_dir):
                datasets[branch]["train"] = ImageFolder(
                    branch_train_dir, transform=self.transform
                )

            if isdir(branch_test_dir):
                datasets[branch]["test"] = ImageFolder(
                    branch_test_dir, transform=self.transform
                )

        # pprint(datasets)
        self.branch_datasets = datasets
        return datasets

    def load_branch_dataloaders(self):
        if self.branch_datasets is None:
            self.branch_datasets = self.load_branch_datasets()

        dataloaders = defaultdict(dict)
        for branch_name, branch_dataset in self.branch_datasets.items():
            dataloaders[branch_name]["train"] = DataLoader(
                branch_dataset["train"], batch_size=16, shuffle=True
            )
            dataloaders[branch_name]["test"] = DataLoader(
                branch_dataset["test"], batch_size=16, shuffle=True
            )

        self.branch_dataloaders = dataloaders
        return dataloaders

    def __repr__(self) -> str:
        # todo: implement this beauty
        pass


if __name__ == "__main__":
    cifar100 = Cifar100()

    dataloaders = cifar100.load_branch_dataloaders()

    for k, v in dataloaders.items():
        print(k)
        print(len(v["train"].dataset.classes))
        pprint(v)
        break
