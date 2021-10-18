import random
import time

from torch.nn.modules.pooling import _AdaptiveAvgPoolNd
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.datasets import mnist
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

seed = 1
torch.manual_seed(seed)
random.seed(seed)

classification = True
conv = False


class ConvNet2D(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet2D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out


model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(),
                      nn.Linear(64, 10))

writer = SummaryWriter()


if __name__ == '__main__':
    epochs = 50
    log_interval = 1000
    batch_size = 32
    lr = 0.01

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = mnist.MNIST(root='./', train=True, transform=data_transform, download=True)
    test_dataset = mnist.MNIST(root='./', train=False, transform=data_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optim = SGD(model.parameters(), lr=lr)
    # optim = AdamW(model.parameters(), lr=lr)
    # scheduler = ExponentialLR(optim, gamma=0.9)
    cost = nn.CrossEntropyLoss() if classification else nn.MSELoss()

    loss_stat = correct = total = 0
    start_time = time.time()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.float()
            if not conv:
                x = torch.flatten(x, start_dim=1)
            # one_hot = F.one_hot(y, num_classes=10).float()
            y_pred = model(x)
            loss = cost(y_pred, y)

            loss_stat += loss.item()
            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]
            if i % log_interval == 0:
                print('epoch: {}, loss: {:.5f}, acc: {}/{} ({:.0f}%)'.format(epoch, loss_stat / log_interval, correct,
                                                                             total, 100. * correct / total))

                writer.add_scalar('Train/Loss', loss_stat / log_interval, epoch * len(train_loader) + i)
                writer.add_scalar('Train/Acc', 100. * correct / total, epoch * len(train_loader) + i)

                loss_stat = correct = total = 0

            optim.zero_grad()
            loss.backward()
            optim.step()
        # scheduler.step()

        correct = total = 0

        for i, (x, y) in enumerate(test_loader):
            x = x.float()
            if not conv:
                x = torch.flatten(x, start_dim=1)
            y_pred = model(x).detach()

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print('epoch: {}, accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total, 100. * correct / total))
        print('time: {}'.format(time.time() - start_time))

        writer.add_scalar('Test/Loss', loss_stat / log_interval, (epoch + 1) * len(train_loader))
        writer.add_scalar('Test/Acc', 100. * correct / total, (epoch + 1) * len(train_loader))

    writer.flush()
