from pprint import pprint
from time import clock_settime

import numpy as np
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

    wikiart_trainloaders = load_wikiart_dataloaders(split="train")
    class_to_idx = wikiart_trainloaders["artists"].dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # pprint(class_to_idx)

    branch_names, test_loaders = [], []
    for k, v in wikiart_trainloaders.items():
        branch_names.append(k)
        test_loaders.append(v)

    # initilize model, optimizer, criterion
    artnet = load_artnet(resume=True)
    artnet.to(device)

    features = np.zeros((9000, 20))

    running_acc = 0.0
    for i, data in tqdm(enumerate(zip(*test_loaders))):

        for branch_name, branch_batch in zip(branch_names, data):
            if branch_name != "artists":
                continue
            images, labels = branch_batch
            images, labels = images.to(device), labels.to(device)
            print(torch.max(images), torch.min(images))

            preds = artnet(images, branch_name)
            print(preds.size())
            pred_idx = torch.argmax(preds, dim=1).detach().numpy()
            print(pred_idx, labels)
            print(pred_idx.shape)
            running_acc += sum(torch.argmax(preds, dim=1) == labels)
            # print(torch.argmax(preds, dim=1))

            break
        break
    print(f"Accuracy: {running_acc/9000:.3f}")
