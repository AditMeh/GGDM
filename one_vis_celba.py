import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from torch.optim import Adam
from unet_labml import UNet
from torchvision.utils import save_image

import torchvision.transforms.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from celeba import create_dataloaders
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import imageio
import os
import tqdm
import argparse

import clip
# Utility functions
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def prompt_helper(input_image, prompt):

    test_image = input_image
    test_image.requires_grad = True
    test_image.retain_grad()



    image = test_image

    upsample = torch.nn.Upsample(size=(224, 224), mode="bicubic")

    image = upsample(image)
    # Normalize
    min_val = torch.min(image)
    max_val = torch.max(image)
    image = (image - min_val) / (max_val-min_val)

    # Normalize with imagenet

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    mean = mean[None, ..., None, None]
    mean = torch.broadcast_to(mean, image.shape)

    var = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    var = var[None, ..., None, None]
    var = torch.broadcast_to(var, image.shape)

    image = (image - mean) / var


    text = text_tokens = clip.tokenize([prompt]).to(device)


    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(image_features, text_features)
    similarity = torch.sum(similarity)

    grad = torch.autograd.grad(similarity, test_image)[0]


    
    return similarity, grad
 
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = torch.sum(text_features @ image_features.T)


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



def sample_chain(net, T, beta_min, beta_max, img_shape, device, prompt, idx):
    hparams = compute_schedule(T, beta_min, beta_max)
    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    for param in net.parameters():
        param.requires_grad = False

    seed = torch.randn(1, *img_shape).to(device=device)

    for i in tqdm.tqdm(range(T, 0, -1)):

        z = torch.randn(1, *img_shape).to(device=device)
        ts = torch.ones(1).to(device) * i

        pred_eps =  net(seed, (ts/T).float())
        term1 = hparams["oneover_sqrta"][i-1]
        term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * pred_eps)
        term3 = z * hparams["var_t"][i-1]

        tweedle = (1 / hparams["sqrt_alpha_bar"][i-1]) * (seed - (hparams["sqrt_one_minus_alpha_bar"][i-1]) *  pred_eps)

        seed = term1 * term2 + term3 if i > 1 else term1 * term2
        # print(torch.min(seed),torch.max(seed))
        loss, grad = prompt_helper(tweedle.detach().clone(), prompt)
        # print(loss, torch.norm(grad))
        seed = seed + 10*grad
    save_image(rescale(seed), f'celba/{prompt}_{idx}.png')

    return seed



def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    # prompt_helper()
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help = "Path to your input image", type=str)
    args = parser.parse_args()

    prompt = args.prompt

    print(prompt)
    
    beta_min, beta_max = 1e-4, 0.02
    T = 1000

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    net = UNet(image_channels = 3, n_blocks = 2, is_attn = (False, False, True, True)).to(device)
    net.load_state_dict(torch.load("celeba_model.pt", map_location=device))
    net.eval()
    train_ = create_dataloaders(1)

    sample = next(iter(train_))

    for i in range(20):
        ret = sample_chain(net, T, beta_min, beta_max, sample.shape[1:], device, args.prompt, i)

    # import cv2
    # images = []
    # for filename in reversed(sorted(os.listdir("./celba/"), key = lambda i: int(i.split(".")[0]))):
    #     pred = cv2.resize(imageio.imread("./celba/" + filename), (128, 128))
    #     tweedle = cv2.resize(imageio.imread("./celba_tweedle/tweedle_" + filename), (128, 128))
    #     zero_img  = np.zeros((128, 256, 3), dtype = np.uint8)
    #     zero_img[:, :128, :] = pred
    #     zero_img[:, 128:, :] = tweedle
    #     images.append(zero_img)

    # take last 300 images of the markov chain. Append the final generated image 20 times for visual clarity. 
    # imageio.mimsave("movie.gif", images[0:] + [images[-1] for _ in range(250)], fps= 60)
    # torch.save(net.state_dict(), "celeba_model.pt")
