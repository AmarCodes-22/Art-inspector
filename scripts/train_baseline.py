from src.data.cifar100 import Cifar100
from src.models.baseline import Baseline


def load_cifar100():
    cifar100 = Cifar100()
    # todo: use load_branch_dataloaders here
    dataloaders = cifar100.load_branch_dataloaders()

    return dataloaders


def load_baseline(branch_dataloaders: dict):
    resnet_layers = [1, 1, 1, 1]
    model = Baseline(stem_layers=resnet_layers)

    stem_out_size = model.stem.get_output_size(input_size=(16, 3, 32, 32))

    # make branch_configs used to initialize branches
    branch_configs = {}
    for branch_name, branch_dataloader in branch_dataloaders.items():
        branch_configs[branch_name] = tuple(
            (stem_out_size, len(branch_dataloader["train"].dataset.classes))
        )

    model.initialize_branches(branch_configs=branch_configs)
    return model


# todo: use argparse here
if __name__ == "__main__":
    cifar_branch_dataloaders = load_cifar100()

    baseline = load_baseline(branch_dataloaders=cifar_branch_dataloaders)
    baseline.print_model_summary(input_size=(16, 3, 32, 32))
