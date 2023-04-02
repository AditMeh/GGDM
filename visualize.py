import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from unet3d.model import UNet3D
import binvox_rw
from shapenet_dataloader import ShapeNetVox

from train_shapenet import compute_schedule, generate_img


def plot_single_voxel(voxel, save_path):
    f, ax = plt.subplots(1, 1)

    colors = np.empty(voxel.shape, dtype=object)
    colors[:, :, :] = 'blue'

    ax.add_subplot(projection='3d')
    ax.voxels(voxel, facecolors=colors, edgecolor='k')
    f.savefig(f'{save_path}.png')


def visualize_voxel():
    T = 128
    beta_min, beta_max = 1e-4, 0.02

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    data_shape = [1, 32, 32, 32]

    train_loader = torch.utils.data.DataLoader(
        ShapeNetVox(), batch_size=1, shuffle=True)

    net = UNet3D(1, 1, final_sigmoid=False, f_maps=[32, 64], layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False).to(device)

    net.load_state_dict(torch.load("diffusion_shapenetcore_model.pt"))
    net.eval()

    hparams = compute_schedule(T, beta_min, beta_max)

    for key, value in hparams.items():
        hparams[key] = hparams[key].to(device)

    result = generate_img(net, T, data_shape, hparams, device)

    final = torch.squeeze(result).cpu().detach().numpy()

    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    final = rescale(final)

    colors = np.empty(final.shape, dtype=object)
    colors[:, :, :] = 'blue'


    for t in np.linspace(0.5, 0.9, 40):
        thresh = (final >= t).astype(np.uint8)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(thresh, facecolors=colors, edgecolor=None)
        plt.savefig(f'{t}_vox.png')


def plot_reverse_diffusion(voxels, save_path):
    raise NotImplementedError


if __name__ == "__main__":
    visualize_voxel()
