import torch
import torch.nn as nn
from unet3d.model import UNet3D
import binvox_rw
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam

import torchvision.transforms.transforms as transforms
from torch.utils.data import DataLoader

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

def generate_img(net, T, img_shape, hparams):
    net.eval()
    with torch.no_grad():
        seed = torch.randn(1, *img_shape).to(device=device)
        for i in tqdm.tqdm(range(T, 0, -1)):
            z = torch.randn(1, *img_shape).to(device=device)
            ts = torch.ones(1).to(device) * i


            term1 = hparams["oneover_sqrta"][i-1]
            term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * net(seed))
            term3 = z * hparams["var_t"][i-1]

            seed = term1 * term2 + term3 if i > 1 else term1 * term2

        return seed


def train(net, x, epochs, T, beta_min, beta_max, img_shape, device):
    hparams = compute_schedule(T, beta_min, beta_max)

    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    loss = nn.MSELoss()
    optimizer = Adam(params=net.parameters(), lr=0.0005)

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        net.train()

        ts = torch.randint(1, T + 1, (x.shape[0],)).to(device)

        eps = torch.randn(*x.shape).to(device=device)

        # Forward pass through model
        x_pass = hparams["sqrt_alpha_bar"][ts - 1][..., None, None, None, None] * x + \
            hparams['sqrt_one_minus_alpha_bar'][ts - 1][..., None, None, None, None] * eps

        pred = net(x_pass) #, ((ts/T).float()))

        # print(torch.norm(pred))
        train_loss = loss(pred, eps)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print(f'epoch: {epoch}, loss = {train_loss.item()}')

        # sampled_img = generate_img(net, T, img_shape, hparams)
        
        # f, ax = plt.subplots(1, 1)

        # ax.imshow((torch.squeeze(sampled_img, dim= 0 ).permute(1,2,0)).detach().cpu().numpy(), cmap="gray")
        # ax.set_axis_off()
        # f.savefig(f'samples/{epoch}.png')
        # f.clear()
        # f.clf()
        # torch.save(net.state_dict(), "celeba_model.pt")


    return net


def sample_chain(net, T, beta_min, beta_max, img_shape, device, ):
    hparams = compute_schedule(T, beta_min, beta_max)
    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)
    seed = torch.randn(3, *img_shape).to(device=device)
    chain_samples = [seed]
    for i in range(T, 0, -1):
        z = torch.randn(1, *img_shape).to(device=device)
        ts = torch.ones(1).to(device) * i

        term1 = hparams["oneover_sqrta"][i-1]
        term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * net(seed, (ts/T).float()))
        term3 = z * hparams["var_t"][i-1]

        seed = term1 * term2 + term3 if i > 1 else term1 * term2
        chain_samples.append(seed)
    return seed, chain_samples

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

if __name__ == "__main__":
    # if not os.path.exists("./folder"):
    #     os.mkdir("folder/")
    # if not os.path.exists("./samples"):
    #     os.mkdir("./samples/")
    T = 128
    beta_min, beta_max = 1e-4, 0.02

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))



    data_shape = [1, 128, 128, 128]
    net = UNet3D(1, 1, final_sigmoid=False, f_maps=[32, 64], layer_order='gcr',
                    num_groups=8, num_levels=4, is_segmentation=False).to(device)



    path = "ShapeNetCore.v2/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.solid.binvox"
    file = open(path, 'rb')
    m1 = binvox_rw.read_as_3d_array(file)
    data = np.transpose(m1.data, (0, 2, 1)).astype(np.float32) # [40:50, 40:50, 0:10]
    data = torch.tensor(data)[None, None, ...].to(device)


    train(net, data, 5, T, beta_min, beta_max, data_shape, device)
    torch.save(net.state_dict(), "celeba_model_final.pt")


    hparams = compute_schedule(T, beta_min, beta_max)

    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)


    result = generate_img(net, T, data_shape, hparams)


    final = torch.squeeze(result).cpu().detach().numpy()
    
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    final = rescale(final)

    colors = np.empty(final.shape, dtype=object)
    colors[:, :, :] = 'blue'
    final = (final >= 0.5).astype(np.uint8)


    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(final, facecolors=colors, edgecolor='k')
    plt.savefig("result.png")