from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

from torchvision.utils import save_image
import cv2
device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

def add_noise_and_blur(data):
    """
    Add Gaussian noise and Gaussian blur to the images in data.

    data: The MNIST data

    """
    new_lst = []
    total_num = len(data)

    group_num = 5
    num_per_group = total_num // group_num

    kernel_lst = [(2 * k + 1, 2 * k + 1) for k in range(group_num)]
    std_lst = [0.01 * k for k in range(group_num)]

    for i in range(total_num):
        # if i == 3:
        #     break

        group_id = i // num_per_group
        kernel = kernel_lst[group_id]
        std = std_lst[group_id]
        
        img, label = data[i]
        img = cv2.GaussianBlur(img.detach().numpy(), kernel, cv2.BORDER_DEFAULT)  # add Gaussian blur
        img = torch.tensor(img)
        img = img + torch.normal(0, std, size=img.shape)  # add Gaussian noise
        new_lst.append((img, label))
    
    return new_lst


data_root = '/voyager/projects/aditya/CSC413/gan-vae-pretrained-pytorch/data/mnist'
random.seed(13)
torch.manual_seed(13)
data_train = MNIST(data_root,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()]))
data_test = MNIST(data_root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()]))
data_train = add_noise_and_blur(data_train)
save_image(data_train[0][0], f'out_0.png')
save_image(data_train[12000][0], f'out_1.png')
save_image(data_train[24000][0], f'out_2.png')
save_image(data_train[36000][0], f'out_3.png')
save_image(data_train[48000][0], f'out_4.png')
data_test = add_noise_and_blur(data_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=512, num_workers=8)

net = LeNet5().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)



def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device=device), labels.to(device=device)
        optimizer.zero_grad()
        
        print(images.shape)
        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

#         if i % 10 == 0:
#             print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    with torch.no_grad():
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device=device), labels.to(device=device)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), acc))
    return acc


def train_and_test(epoch):
    print('training...')
    train(epoch)
    acc = test()
    return acc


def main():
    for e in range(1, 13):
        acc = train_and_test(e)
        if e % 2 == 0:
            torch.save(net.state_dict(), f'./lenet_epoch={e}_test_acc={acc:0.3f}.pth')


if __name__ == '__main__':
    main()
