import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from torch.optim import Adam
from unet_labml import UNet

import torchvision.transforms.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from celeba import create_dataloaders

import imageio
import os
import tqdm
# Utility functions


def compute_schedule(T, beta_min, beta_max):
    betas = torch.linspace(beta_min, beta_max, steps=T)
    alphas = 1 - betas

    var_t = torch.sqrt(betas)
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    hparams = {
        "var_t": var_t,
        "alphas": alphas,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "oneover_sqrta": 1/torch.sqrt(alphas),
        "mab_over_sqrtmab": (1-alphas)/sqrt_one_minus_alpha_bar
    }

    return hparams
from torchvision.utils import save_image

def sample_chain(net, T, beta_min, beta_max, img_shape, device):
    hparams = compute_schedule(T, beta_min, beta_max)
    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)


    seed = torch.randn(1, *img_shape).to(device=device)
    with torch.no_grad():
        for i in tqdm.tqdm(range(T, 0, -1)):

            z = torch.randn(1, *img_shape).to(device=device)
            ts = torch.ones(1).to(device) * i

            pred_eps =  net(seed, (ts/T).float())
            term1 = hparams["oneover_sqrta"][i-1]
            term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * pred_eps)
            term3 = z * hparams["var_t"][i-1]

            tweedle = (1 / hparams["sqrt_alpha_bar"][i-1]) * (seed - (hparams["sqrt_one_minus_alpha_bar"][i-1]) *  pred_eps)

            seed = term1 * term2 + term3 if i > 1 else term1 * term2

            save_image(rescale(seed), f'celba/{i}.png')
            save_image(rescale(tweedle), f'celba_tweedle/tweedle_{i}.png')

    return seed



def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    beta_min, beta_max = 1e-4, 0.02
    T = 1000

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    net = UNet(image_channels = 3, n_blocks = 2, is_attn = (False, False, True, True)).to(device)
    net.load_state_dict(torch.load("celeba_model.pt", map_location=device))
    net.eval()
    train_ = create_dataloaders(1)

    sample = next(iter(train_))


    ret = sample_chain(net, T, beta_min, beta_max, sample.shape[1:], device)

    import cv2
    images = []
    for filename in reversed(sorted(os.listdir("./celba/"), key = lambda i: int(i.split(".")[0]))):
        pred = cv2.resize(imageio.imread("./celba/" + filename), (128, 128))
        tweedle = cv2.resize(imageio.imread("./celba_tweedle/tweedle_" + filename), (128, 128))
        zero_img  = np.zeros((128, 256, 3), dtype = np.uint8)
        zero_img[:, :128, :] = pred
        zero_img[:, 128:, :] = tweedle
        images.append(zero_img)

    # take last 300 images of the markov chain. Append the final generated image 20 times for visual clarity. 
    imageio.mimsave("movie.gif", images[0:] + [images[-1] for _ in range(250)], fps= 60)
    # torch.save(net.state_dict(), "celeba_model.pt")
