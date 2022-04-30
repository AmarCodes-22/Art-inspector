import torch
from tqdm import tqdm

from src import ARTNET_CONFIG_FPATH
from src.data.wikiart import Wikiart
from src.models.artnet import ArtNet
from src.utils import load_config


def load_wikiart_dataloaders(split: str):
    wikiart = Wikiart(split=split)
    return wikiart.load_dataloaders()


def load_artnet(config_fpath=ARTNET_CONFIG_FPATH, resume=False):
    config = load_config(ARTNET_CONFIG_FPATH)
    if resume:
        artnet = ArtNet(config_fpath)
        artnet.load_state_dict(
            torch.load(config.TRAINING.RESUME_WEIGHTS, map_location="cpu")
        )
        return artnet
    else:
        return ArtNet(config_fpath)


# todo: implement argparse
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(ARTNET_CONFIG_FPATH)

    wikiart_trainloaders = load_wikiart_dataloaders(split="test")

    branch_names, test_loaders = [], []
    for k, v in wikiart_trainloaders.items():
        branch_names.append(k)
        test_loaders.append(v)

    # initilize model, optimizer, criterion
    artnet = load_artnet(resume=True)
    artnet.to(device)

    running_acc = 0.0
    for i, data in tqdm(enumerate(zip(*test_loaders))):

        for branch_name, branch_batch in zip(branch_names, data):
            images, labels = branch_batch
            images, labels = images.to(device), labels.to(device)

            preds = artnet(images, branch_name)
            running_acc += sum(torch.argmax(preds, dim=1) == labels)

            # break
        # break
    print(f"Accuracy: {running_acc/4987:.3f}")
