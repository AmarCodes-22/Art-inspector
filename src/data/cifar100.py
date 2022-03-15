from os import listdir, getcwd
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Mapping
from os.path import join, isdir

from torchvision.datasets import ImageFolder


class Cifar100:
    def __init__(self, dir_path: str, branch_names: Optional[List[str]] = None) -> None:
        """Initialize Cifar100 dataset

        Parameters
        ----------
        dir_path : str
            Path to the directory where this dataset is located
        branch_names : List[str]
            List of branch names to load the dataset for
        """
        self.dir_path = dir_path

        if branch_names is None:
            self.branch_names = listdir(join(dir_path, "train"))
        else:
            self.valid_branch_names = set(listdir(join(dir_path, "train")))
            for branch_name in branch_names:
                if branch_name not in self.valid_branch_names:
                    print(
                        "{} is not a valid branch name. Should be one of {}".format(
                            branch_name, self.valid_branch_names
                        )
                    )
                break
            self.branch_names = branch_names

    def load_branch_datasets(self) -> Mapping[str, Mapping[str, ImageFolder]]:
        """Load datasets for all branches inside Cifar100

        Returns
        -------
        dict: Mapping[str, Mapping[str, ImageFolder]]
            Dictionary mapping from each branch's train and test to torchvision.datasets.ImageFolder instance
        """
        datasets = defaultdict(dict)

        for i, branch in enumerate(self.branch_names):
            print(f"{i}. Creating dataset for {branch}")

            branch_train_dir = join(self.dir_path, "train", branch)
            branch_test_dir = join(self.dir_path, "test", branch)

            if isdir(branch_train_dir):
                datasets[branch]["train"] = ImageFolder(branch_train_dir)

            if isdir(branch_test_dir):
                datasets[branch]["test"] = ImageFolder(branch_test_dir)

        pprint(datasets)
        return datasets

    def __repr__(self) -> str:
        #todo: implement this
        pass


if __name__ == "__main__":
    cifar100_dir_path = join(Path(getcwd()).parent, "data", "cifar100")
    cifar100 = Cifar100(cifar100_dir_path)

    datasets = cifar100.load_branch_datasets()
