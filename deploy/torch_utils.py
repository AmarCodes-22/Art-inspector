import io
import pickle

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src import ARTNET_CONFIG_FPATH
from src.models.artnet import ArtNet
from src.utils import load_config


def transform_image(img_bytes):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    if isinstance(img_bytes, str):
        img = Image.open(img_bytes)
    else:
        img = Image.open(io.BytesIO(img_bytes))
    return transform(img).unsqueeze_(0).to('cpu')


def load_idx_to_class():
    with open('class_label_map.yaml', 'r') as stream:
        class_to_idx = yaml.safe_load(stream)

    idx_to_class = {v:k for k, v in class_to_idx.items()}

    return idx_to_class


def get_pred(model, image_bytes):
    img_tensor = transform_image(image_bytes)
    model.eval()
    return model(img_tensor, 'artists')


def load_search_features():
    with open('feature_store.pkl', 'rb') as stream:
        feature_store = pickle.load(stream)
    return feature_store


def load_model():
    config = load_config(ARTNET_CONFIG_FPATH)
    artnet = ArtNet(ARTNET_CONFIG_FPATH)
    artnet.load_state_dict(torch.load(
        config.TRAINING.RESUME_WEIGHTS, map_location='cpu'
    ))
    return artnet


def get_top_10(image_bytes):
    model = load_model()
    preds_arr = get_pred(model, image_bytes).detach().numpy()
    feature_store = load_search_features()
    # features_only = np.array(list(feature_store.values()))

    feat_diff = dict()
    for k, v in feature_store.items():
        feat_diff[k] = (preds_arr - v)

    result = []
    sorted_feats = dict(sorted(feat_diff.items(), key=lambda item: np.linalg.norm(item[1])))
    for i, (k, v) in enumerate(sorted_feats.items()):
        result.append(k)
        # print(k, v.shape)

        if i == 9:
            break

    return result


if __name__ == '__main__':
    image_fpath = '''/home/amar/dev/projects/major_project/art_inspector/deploy/subset_dataset/Albrecht Durer/23.jpg'''
    get_top_10(image_fpath)
