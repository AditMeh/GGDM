import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 32, kernel_size=(5, 5))),
            ('relu5', nn.ReLU()),
            ('s6', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c7', nn.Conv2d(32, 120, kernel_size=(5, 5))),
            ('relu7', nn.ReLU()),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(7680, 768)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(768, 84)),
            ('relu7', nn.ReLU()),
            ('f8', nn.Linear(84, 10)),
            ('sig8', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


def load_classifier_model(path='/voyager/projects/aditya/Prior-Conditional-3D-Diffusion/checkpoints/lenet_epoch=12_test_acc=0.989.pth'):
    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    model = LeNet5().to(device=device)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)

    return model