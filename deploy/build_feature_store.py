import os
import pickle

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src import ARTNET_CONFIG_FPATH
from src.models.artnet import ArtNet
from src.utils import load_config


def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    return transform(img).unsqueeze_(0).to('cpu')


def load_idx_to_class():
    with open('class_label_map.yaml', 'r') as stream:
        class_to_idx = yaml.safe_load(stream)

    idx_to_class = {v:k for k, v in class_to_idx.items()}

    return idx_to_class


def get_pred(model, image_fpath):
    img_tensor = transform_image(Image.open(image_fpath))
    model.eval()
    return model(img_tensor, 'artists')


def load_search_features():
    feats = np.load('artists_branch_features.npy')
    return feats


if __name__ == '__main__':
    # load model
    config = load_config(ARTNET_CONFIG_FPATH)
    artnet = ArtNet(ARTNET_CONFIG_FPATH)
    artnet.load_state_dict(torch.load(
        config.TRAINING.RESUME_WEIGHTS, map_location='cpu'
    ))

    feature_store = dict()
    # load all images one by one
    for root, dirs, files in os.walk('subset_dataset'):
        for fname in files:
            pred = get_pred(artnet, os.path.join(root, fname))
            feature_store[os.path.join(root,fname)] = np.squeeze(pred.detach().numpy())

    with open('feature_store.pkl', 'wb') as stream:
        pickle.dump(feature_store, stream)

    print(len(feature_store))
