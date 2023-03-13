import glob
import torch
import binvox_rw
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np


class LoadVoxel(object):
    def __init__(self) -> None:
        pass

    def __call__(self, file_path):
        file = open(file_path, 'rb')
        m1 = binvox_rw.read_as_3d_array(file)
        data = np.transpose(m1.data, (0, 2, 1)).astype(np.float32)
        data = data[None, ...]  # add channels
        file.close()

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ShapeNetVox(torch.utils.data.Dataset):
    def __init__(self):

        self.voxels = glob.glob(
            "ShapeNetVox32/02691156" + '/**/*.binvox', recursive=True)
        self.transforms = transforms.Compose([LoadVoxel(), torch.tensor])

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        return self.transforms((voxel))


if __name__ == "__main__":
    d = ShapeNetVox()
    train = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True)
