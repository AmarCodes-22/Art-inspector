from pprint import pprint

from torch import nn
from torch import optim

from src.data.cifar100 import Cifar100
from src.models.baseline import Baseline


def load_cifar100():
    cifar100 = Cifar100()
    dataloaders = cifar100.load_branch_dataloaders()
    return dataloaders


def load_baseline(branch_dataloaders: dict):
    resnet_layers = [1, 1, 1, 1]
    model = Baseline(stem_layers=resnet_layers)

    stem_out_size = model.stem.get_output_size(input_size=(16, 3, 32, 32))

    # make branch_configs used to initialize branches
    branch_configs = dict()
    for branch_name, branch_dataloader in branch_dataloaders.items():
        branch_configs[branch_name] = tuple(
            (stem_out_size, len(branch_dataloader["train"].dataset.classes))
        )

    model.initialize_branches(branch_configs=branch_configs)
    return model


# todo: use argparse here
if __name__ == "__main__":
    cifar_branch_dataloaders = load_cifar100()
    # pprint(cifar_branch_dataloaders)

    baseline = load_baseline(branch_dataloaders=cifar_branch_dataloaders)
    # baseline.print_model_summary(input_size=(16, 3, 32, 32))

    branches = list(cifar_branch_dataloaders.keys())

    train_loaders = []
    test_loaders = []
    for branch in branches:
        dataloaders = cifar_branch_dataloaders[branch]

        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(baseline.parameters(), lr=0.0005, momentum=0.9)

    for epoch in range(2):
        for i, data in enumerate(zip(*train_loaders)):
            running_loss = 0.0
            for branch_num, (branch_name, branch_batch) in enumerate(
                zip(branches, data)
            ):
                images, labels = branch_batch
                optimizer.zero_grad()

                # todo: move forward, backward, optimize in func train_step
                # forward
                outputs = baseline(images, branch_name)

                # backward
                loss = criterion(outputs, labels)
                loss.backward()

                # optimize
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch: {epoch + 1}, Batch: {i+1}, Loss: {running_loss:.3f}")
            running_loss = 0.0

    print("Finished training")
