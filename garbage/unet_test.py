import torch
import torch.nn as nn
from unet3d.model import UNet3D
import binvox_rw
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


net = UNet3D(1, 1, final_sigmoid=True, f_maps=[32, 64], layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False)

path = "ShapeNetCore.v2/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.solid.binvox"

file = open(path, 'rb')
m1 = binvox_rw.read_as_3d_array(file)
data = np.transpose(m1.data, (0, 2, 1)).astype(np.float32) # [40:50, 40:50, 0:10]




data = torch.tensor(data)[None, None, ...]

mse = MSELoss()


epochs = 200
device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
net = UNet3D(1, 1, final_sigmoid=True, f_maps=[32, 64, 128, 256, 512], layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False).to(device)
data = data.to(device)
optimizer = Adam(params=net.parameters(), lr=0.0005)
for epoch in range(1, epochs +1):
    pred = net(data)

    loss = mse(data, pred)

    net.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)

