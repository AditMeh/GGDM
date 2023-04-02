import torch
import torch.nn as nn
import numpy as np

import tqdm
import os
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class CelebA(torch.utils.data.Dataset):
    def __init__(self, image_fps):

        self.images = image_fps
        self.transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx])
        return self.transforms((im))


def create_dataloaders(batch_size):
    base_path = "data/celeba/img_align_celeba/img_align_celeba"
    fps = [os.path.join(base_path, pth)
           for pth in os.listdir(os.path.join(base_path))]

    # insert logic for creating the dataloaders
    train = torch.utils.data.DataLoader(
        CelebA(fps),
        batch_size=batch_size, shuffle=True)

    return train


if __name__ == "__main__":
    a, b = create_dataloaders(**{"batch_size": 1, "split": 0.70})
    for i in tqdm.tqdm(a):
        assert ((i.shape[2] == 218) and (i.shape[3] == 178))
    for i in tqdm.tqdm(b):
        assert ((i.shape[2] == 218) and (i.shape[3] == 178))
    print(len(a), len(b))
