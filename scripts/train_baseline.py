from pprint import pprint
from typing import Optional

from src.data.cifar100 import Cifar100
from src.models.baseline import Baseline

# Instantiate dataset
def load_cifar100():
    cifar100 = Cifar100()
    datasets = cifar100.load_branch_datasets()

    return datasets


# Instantiate Baseline
def load_baseline(branch_datasets: dict):
    resnet_layers = [1, 1, 1, 1]
    model = Baseline(stem_layers=resnet_layers)

    stem_out_size = model.stem.get_output_size(input_size=(16, 3, 32, 32))

    branches = {}
    for k, v in branch_datasets.items():
        branches[k] = tuple((stem_out_size, len(v["train"].classes)))

    model.initialize_branches(branches=branches)
    return model


# todo: use argparse here
if __name__ == "__main__":
    cifar_branch_datasets = load_cifar100()

    baseline = load_baseline(branch_datasets=cifar_branch_datasets)
    baseline.print_model_summary(input_size=(16, 3, 32, 32))
