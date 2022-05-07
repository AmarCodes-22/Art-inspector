from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src import ARTNET_CONFIG_FPATH
from src.models.artnet import ArtNet
from src.utils import load_config

# make dataset
pred_images_folder = (
    "/home/amar/dev/projects/major_project/art_inspector/scripts/subset_dataset"
)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
preds_dataset = ImageFolder(pred_images_folder, transform=transform)
class_to_idx = preds_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
pprint(idx_to_class)

# make model
config = load_config(ARTNET_CONFIG_FPATH)
artnet = ArtNet(ARTNET_CONFIG_FPATH)
artnet.load_state_dict(torch.load(config.TRAINING.RESUME_WEIGHTS, map_location="cpu"))

dataloader = DataLoader(preds_dataset, batch_size=1, shuffle=True)
meh = dataloader.dataset.samples
# print(meh[:10])

feats_place_holder = np.zeros((400, 20))
feats_dict = dict()
for i, batch in enumerate(dataloader):
    images, labels = batch[0].to("cpu"), batch[1].to("cpu")
    preds = artnet(images, "artists")

    # feats_dict[]
    # print(preds.shape)

    # batch_size = len(batch[0])
    # feats_place_holder[
    #     i * batch_size : i * batch_size + batch_size
    # ] = preds.detach().numpy()

    # break

print(feats_place_holder.shape)
np.save("artists_branch_features.npy", feats_place_holder)

# for images, labels in dataloader:
#     images, labels = images.to('cpu'), labels.to('cpu')
#     preds = artnet(images, 'artists')
#     print(preds.shape)
#     # preds = torch.argmax(preds, dim=1)

#     # print(preds, labels)
#     # print(sum(preds == labels))

#     break

# print(preds_dataset.classes)
