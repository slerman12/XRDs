import argparse
import glob
import random
import time

from torch.optim.lr_scheduler import ExponentialLR
from torchvision.datasets import mnist
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from data import XRDData
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', default="runs", type=str, help='logging directory')
parser.add_argument('--num-workers', default=0, type=int, help='number data loading workers')
args = parser.parse_args()


seed = 1
torch.manual_seed(seed)
random.seed(seed)

classification = True
conv = False


class ConvNet1D(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(1))
        self.fc = nn.Linear(14400, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out


class ConvNet2D(nn.Module):
    def __init__(self, num_classes=6):
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


# model = nn.Sequential(nn.Linear(1800, 512), nn.ReLU(),
#                       nn.Linear(512, 256), nn.ReLU(),
#                       nn.Linear(256, 128), nn.ReLU(),
#                       nn.Linear(128, 64), nn.ReLU(),
#                       nn.Linear(64, 7))

model = ConvNet1D()
conv = True

# model = nn.Sequential(nn.Linear(1800, 7))

writer = SummaryWriter(log_dir=args.log_dir)


if __name__ == '__main__':
    epochs = 1000
    log_interval = 1000
    batch_size = 32
    lr = 0.01

    saved = glob.glob("./*.pt")
    train_test_split = 0.9
    if './train_loader.pt' in saved:
        print("loading train...")
        train_loader = torch.load('train_loader.pt')
        print("done")
    else:
        print("parsing train...")
        train_dataset = XRDData(root='xrd_data/47049', train=True, train_test_split=train_test_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        torch.save(train_loader, 'train_loader.pt')
        print("done")

    if './test_loader.pt' in saved:
        print("loading test...")
        test_loader = torch.load('test_loader.pt')
        print("done")
    else:
        print("parsing test...")
        test_dataset = XRDData(root='xrd_data/47049', train=False, train_test_split=train_test_split)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        torch.save(test_loader, 'test_loader.pt')
        print("done")

    optim = SGD(model.parameters(), lr=lr)
    # optim = AdamW(model.parameters(), lr=lr)
    # scheduler = ExponentialLR(optim, gamma=0.9)
    cost = nn.CrossEntropyLoss() if classification else nn.MSELoss()

    loss_stat = correct = total = 0
    start_time = time.time()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.float()
            # print(torch.isnan(x).sum())
            x[torch.isnan(x)] = 0
            if not torch.nonzero(x).any():
                continue
            if torch.isinf(x).any():
                continue
            if torch.isnan(x).any():
                continue
            x = torch.flatten(x, start_dim=1)
            if conv:
                x = x.unsqueeze(1)
                assert x.shape[1] == 1
                assert x.shape[2] == 1800
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
            if not torch.nonzero(x).any():
                continue
            if torch.isinf(x).any():
                continue
            if torch.isnan(x).any():
                continue
            x = torch.flatten(x, start_dim=1)
            if conv:
                x = x.unsqueeze(1)
                assert x.shape[1] == 1
                assert x.shape[2] == 1800
            y_pred = model(x).detach()

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print('epoch: {}, accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total, 100. * correct / total))
        print('time: {}'.format(time.time() - start_time))

        writer.add_scalar('Test/Loss', loss_stat / log_interval, (epoch + 1) * len(train_loader))
        writer.add_scalar('Test/Acc', 100. * correct / total, (epoch + 1) * len(train_loader))

    writer.flush()
