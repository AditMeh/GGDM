import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import TensorDataset
# from unet_sandbox import NaiveUnet
from models.unet_labml import UNet

from dataloaders.mnist import create_dataloaders

import torchvision.transforms.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from models.classifier import load_classifier_model

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


def generate_img(net, T, img_shape, hparams):
    net.eval()
    with torch.no_grad():
        seed = torch.randn(1, *img_shape).to(device=device)
        for i in range(T, 0, -1):
            z = torch.randn(1, *img_shape).to(device=device)
            ts = torch.ones(1).to(device) * i

            term1 = hparams["oneover_sqrta"][i-1]
            term2 = seed - (hparams["mab_over_sqrtmab"]
                            [i-1] * net(seed, (ts/T).float()))
            term3 = z * hparams["var_t"][i-1]

            seed = term1 * term2 + term3 if i > 1 else term1 * term2

        return seed


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, sizes, img_shape):
        super().__init__()

        self.encoder_layers, self.decoder_layers = create_network(
            sizes, *img_shape)
        self.posenc = TimeSiren(sizes[-1])

    def forward(self, x, t):
        forward_cache = []

        # Forward pass in encoder
        for layer in self.encoder_layers:
            x = layer(x)
            forward_cache.append(x)

        x = self.posenc(t)[..., None, None] + x  # Add time embedding

        # Forward pass in decoder
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x = layer(x)
            else:
                enc_feat_map = forward_cache[-(i+1)]

                # x = torch.cat([x, enc_feat_map], dim=1)
                x = layer(x)
        return x


def block(sin, sout): return nn.Sequential(
    *[nn.Conv2d(sin, sout, 3, 2, 1), nn.BatchNorm2d(sout), nn.ReLU()])


def upsample_block(sin, sout): return nn.Sequential(
    *[nn.ConvTranspose2d(sin, sout, 3, 2, 1, 1), nn.BatchNorm2d(sout), nn.ReLU()])


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class AltUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down0 = block(1, 128)
        self.down1 = block(128, 512)
        self.down2 = block(512, 1024)
        self.down3 = block(1024, 2048)

        self.up0 = upsample_block(2048, 1024)
        self.up1 = upsample_block(2048, 512)
        self.up2 = upsample_block(1024, 128)
        self.up3 = upsample_block(256, 1)

        self.timenc = TimeSiren(2048)

    def forward(self, x, t):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        down3 = down3 + self.timenc(t)[..., None, None]

        up0 = self.up0(down3)
        up1 = self.up1(torch.cat([up0, down2], dim=1))
        up2 = self.up2(torch.cat([up1, down1], dim=1))
        up3 = self.up3(torch.cat([up2, down0], dim=1))

        return up3


def train(net, train_loader, epochs, T, beta_min, beta_max, img_shape, device):
    hparams = compute_schedule(T, beta_min, beta_max)

    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    loss = nn.MSELoss()
    optimizer = Adam(params=net.parameters(), lr=1e-5)

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        for x, _ in tqdm.tqdm(train_loader):
            net.train()
            x = x.to(device=device)
            ts = torch.randint(1, T + 1, (x.shape[0],)).to(device)

            eps = torch.randn(*x.shape).to(device=device)

            # Forward pass through model
            x_pass = hparams["sqrt_alpha_bar"][ts - 1][..., None, None, None] * x + \
                hparams['sqrt_one_minus_alpha_bar'][ts -
                                                    1][..., None, None, None] * eps

            pred = net(x_pass, ((ts/T).float()))
            # print(torch.norm(pred))
            train_loss = loss(pred, eps)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss = {train_loss.item()}')

        sampled_img = generate_img(net, T, img_shape, hparams)

        f, ax = plt.subplots(1, 1)

        ax.imshow((torch.squeeze(sampled_img, dim=0).permute(
            1, 2, 0)).detach().cpu().numpy(), cmap="gray")
        ax.set_axis_off()
        f.savefig(f'samples/{epoch}.png')
        f.clear()
        f.clf()

    return net


def sample_chain(net, classifier, T, beta_min, beta_max, img_shape, device):
    hparams = compute_schedule(T, beta_min, beta_max)
    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    seed = torch.randn(1, *img_shape).to(device=device)
    chain_samples = [seed]

    for param in net.parameters():
        param.requires_grad = False

    target_class = 0
    for i in range(T, 0, -1):
        print(i)
        z = torch.randn(1, *img_shape).to(device=device)
        ts = torch.ones(1).to(device) * i

        pred_eps = net(seed, (ts/T).float())

        term1 = hparams["oneover_sqrta"][i-1]
        term2 = seed - (hparams["mab_over_sqrtmab"][i-1] * pred_eps)
        term3 = z * hparams["var_t"][i-1]

        tweedle = (1 / hparams["sqrt_alpha_bar"][i-1]) * (seed -
                                                          (hparams["sqrt_one_minus_alpha_bar"][i-1]) * pred_eps)

        tweedle.requires_grad = True
        tweedle.retain_grad()

        softmax_distribution = torch.exp(classifier(tweedle))

        one_hot = torch.eye(10)[target_class].to(device)

        loss = -torch.sum(one_hot * softmax_distribution)

        print(loss)
        loss.backward(retain_graph=True)

        gradient = tweedle.grad

        save_image(seed, f'celba/{i}.png')
        save_image(tweedle, f'celba_tweedle/tweedle_{i}.png')

        # Compute x_t-1 like in DDPM
        seed = term1 * term2 + term3 if i > 1 else term1 * term2

        # Compute x_t-1 = x_t-1 - gradient of loss w.r.t x_t-1


        seed = seed - gradient

        # chain_samples.append(seed)

        classifier.zero_grad()
        net.zero_grad()
    return seed, chain_samples


if __name__ == "__main__":
    if not os.path.exists("./folder"):
        os.mkdir("folder/")
    if not os.path.exists("./samples"):
        os.mkdir("./samples/")
    T = 1000
    beta_min, beta_max = 1e-4, 0.02

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    tf = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    train_ = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    sample = next(iter(train_))[0]

    # Build a static graph
    # net = DiffusionModel([1, 128, 512, 1024, 2048], sample.shape[1:]).to(device=device)
    # net = NaiveUnet(1, 1, n_feat=128).to(device=device)
    # net = AltUnet().to(device=device)
    net = UNet(image_channels=1, n_blocks=1, is_attn=(
        False, False, True, True)).to(device)
    net.load_state_dict(torch.load("checkpoints/mnist_model.pt", map_location=device))
    net.eval()
    classifier = load_classifier_model()
    classifier.eval()

    # train(net, train_, 100, T, beta_min, beta_max,  sample.shape[1:], device)
    _, ret = sample_chain(net, classifier, T, beta_min,
                          beta_max, sample.shape[1:], device)

    import cv2

    images = []
    for filename in reversed(sorted(os.listdir("./celba/"), key=lambda i: int(i.split(".")[0]))):
        pred = cv2.resize(imageio.imread("./celba/" + filename), (128, 128))
        tweedle = cv2.resize(imageio.imread(
            "./celba_tweedle/tweedle_" + filename), (128, 128))
        zero_img = np.zeros((128, 256, 3), dtype=np.uint8)
        zero_img[:, :128, :] = pred
        zero_img[:, 128:, :] = tweedle
        images.append(zero_img)

    # take last 300 images of the markov chain. Append the final generated image 20 times for visual clarity.
    imageio.mimsave("movie.gif", images[0:] +
                    [images[-1] for _ in range(250)], fps=60)
