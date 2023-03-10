import os
import sys

import numpy as np
import pytorch3d
import torch
import matplotlib.pyplot as plt

import binvox_rw
import os

base = 'ShapeNetCore.v2/02691156/'
import tqdm
# mean = np.zeros((128, 128, 128))

# total = 0
# for file in tqdm.tqdm(os.listdir("ShapeNetCore.v2/02691156/")):
#     f = open(os.path.join(base, file, 'models/model_normalized.solid.binvox'), 'rb')
#     m1 = binvox_rw.read_as_3d_array(f)
#     data = np.transpose(m1.data, (0, 2, 1)).astype(np.float32) # [40:50, 40:50, 0:10]
#     mean += data
#     total += 1



# colors = np.empty(mean.shape, dtype=object)
# colors[:, :, :] = 'blue'

# mean = mean/total
# print(np.unique(mean))
# mean = (mean > 0).astype(np.uint8)

# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(mean, facecolors=colors, edgecolor='k')

path = "/voyager/projects/aditya/Prior-Conditional-3D-Diffusion/ShapeNetVox32/02691156/5f11d3dd6ccacf92de7a468bfa758b34/model.binvox"
file = open(path, 'rb')
m1 = binvox_rw.read_as_3d_array(file)
data = np.transpose(m1.data, (0, 2, 1)).astype(np.uint8) # [40:50, 40:50, 0:10]

colors = np.empty(data.shape, dtype=object)
colors[:, :, :] = 'blue'

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(data, facecolors=colors, edgecolor='k')
plt.savefig("new.png")

# from pytorch3d.datasets import (
#     R2N2,
#     ShapeNetCore,
#     collate_batched_meshes,
#     render_cubified_voxels,
# )

# from pytorch3d.structures import Meshes
# from torch.utils.data import DataLoader

# # add path for demo utils functions 
# import sys
# import os

# # Setup
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")
    
# SHAPENET_PATH = "./ShapeNetCore.v2/"
# shapenet_dataset = ShapeNetCore(SHAPENET_PATH)


# print(next(iter(shapenet_dataset)).shape)