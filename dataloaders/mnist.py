import torchvision.datasets as datasets
import torch
from torchvision.transforms.transforms import ToTensor, Compose, Lambda
from torch.nn.functional import dropout


class DropoutPixelsTransform(object):
    def __init__(self, prob) -> None:
        self.prob = prob

    def __call__(self, tensor):
        return dropout(tensor, self.prob)  # inplace dropout

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BinarizeTransform(object):
    def __init__(self) -> None:
        pass

    def __call__(self, tensor):
        return (~(tensor == 0)).float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class UnflattenImageTransform(object):
    def __init__(self, height, width, channels) -> None:
        self.height = height
        self.width = width
        self.channels = channels

    def __call__(self, tensor):
        batch_size = tensor.shape[0]
        return torch.reshape(tensor, (self.channels, self.width, self.height))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def create_dataloaders(batch_size, tanh_normalize=False):

    transforms = [ToTensor(), Lambda(lambda x: torch.flatten(x))]

    if tanh_normalize:
        transforms.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    transforms.append(UnflattenImageTransform(28, 28, 1))
    transforms = Compose(transforms)

    train_dataset = datasets.MNIST(
        root='./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

    val_dataset = datasets.MNIST(
        root='./data/', train=False, download=True, transform=transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    return train_loader, val_loader
