from torch import nn, optim

from src.data.wikiart import Wikiart
from src.models import ARTNET_CONFIG_FPATH
from src.models.artnet import ArtNet


def load_wikiart_dataloaders(split: str):
    wikiart = Wikiart(split=split)
    return wikiart.load_dataloaders()


def load_artnet(config_fpath=ARTNET_CONFIG_FPATH):
    return ArtNet(config_fpath)


def train_step(branch_name, inputs, labels, model, optimizer, criterion):
    optimizer.zero_grad()

    # forward
    preds = model(inputs, branch_name)

    # backward
    loss = criterion(preds, labels)
    loss.backward()

    # optimize
    optimizer.step()

    return preds, loss


if __name__ == "__main__":
    wikiart_trainloaders = load_wikiart_dataloaders(split="train")

    branch_names, train_loaders = [], []
    for k, v in wikiart_trainloaders.items():
        branch_names.append(k)
        train_loaders.append(v)

    # initilize model, optimizer, criterion
    artnet = load_artnet()
    optimizer = optim.SGD(artnet.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # number of times to go through the dataset
    for epoch in range(2):

        # data is a tuple of 3 batches, one from each dataset
        for i, data in enumerate(zip(*train_loaders)):
            # todo: this is better named batch loss because printing it out every batch
            running_loss = 0.0

            # loop over the 3 batches wrt branch
            for branch_name, branch_batch in zip(branch_names, data):
                images, labels = branch_batch

                preds, loss = train_step(
                    branch_name, images, labels, artnet, optimizer, criterion
                )
                running_loss += loss.item()
                # todo: get accuracy
                # print(preds.shape, labels.shape)
                # print(sum(preds == labels))

                # break

            # print stats every batch
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss:.3f}")
            running_loss = 0.0

            # break
        # break
    print("Finished training")
