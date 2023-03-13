import numpy as np
import matplotlib.pyplot as plt

def plot_single_voxel(voxel, save_path):
    f, ax = plt.subplots(1,1)

    colors = np.empty(voxel.shape, dtype=object)
    colors[:, :, :] = 'blue'

    ax.add_subplot(projection='3d')
    ax.voxels(voxel, facecolors=colors, edgecolor='k')
    f.savefig(f'{save_path}.png')


def plot_reverse_diffusion(voxels, save_path):
    raise NotImplementedError
