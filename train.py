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
from torchsummary import summary

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', default="runs", type=str, help='logging directory')
parser.add_argument('--name', default="dnn", type=str, help='logging directory')
parser.add_argument('--num-workers', default=0, type=int, help='number data loading workers')
args = parser.parse_args()


seed = 1
torch.manual_seed(seed)
random.seed(seed)

classification = True
conv = False
paper = False


class ConvNet1DPaper(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet1DPaper, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=(100, 1, 1), stride=5),
            nn.AvgPool1d(kernel_size=(3, 1, 1), stride=2),
            nn.Conv1d(1, 80, kernel_size=(50, 1, 80), stride=5),
            nn.AvgPool1d(kernel_size=(3, 1, 1), stride=1),
            nn.Conv1d(1, 80, kernel_size=(25, 1, 80), stride=2),
            nn.AvgPool1d(kernel_size=(3, 1, 1), stride=1),
            nn.Flatten(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(880, 700),
            nn.Linear(700, 70),
            nn.Linear(70, num_classes))

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out


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


if "dnn" in args.name:
    model = nn.Sequential(nn.Linear(1800, 512), nn.ReLU(),
                          nn.Linear(512, 256), nn.ReLU(),
                          nn.Linear(256, 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(),
                          nn.Linear(64, 7))
elif args.name == "cnn":
    model = ConvNet1D()
    conv = True
elif args.name == "cnnp":
    model = ConvNet1DPaper()
    conv = True
    paper = True
elif "logreg" in args.name:
    model = nn.Sequential(nn.Linear(1800, 7))
else:
    assert False

writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.name}")

if conv:
    if paper:
        summary(model, (1, 1800))
    else:
        summary(model, (1, 1800))


if __name__ == '__main__':
    epochs = 1000
    log_interval = 1000
    batch_size = 32
    lr = 0.01

    saved = glob.glob("./*.pt")
    train_test_split = 0.9
    if './train_loader.pt' in saved and False:
        print("loading train...")
        train_loader = torch.load('train_loader.pt')
        print("done")
    else:
        print("parsing train...")
        train_dataset = XRDData(root='xrd_data/47049', train=True, train_test_split=train_test_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        # torch.save(train_loader, 'train_loader.pt')
        print("done")

    if './test_loader.pt' in saved and False:
        print("loading test...")
        test_loader = torch.load('test_loader.pt')
        print("done")
    else:
        print("parsing test...")
        test_dataset = XRDData(root='xrd_data/47049', train=False, train_test_split=train_test_split)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # torch.save(test_loader, 'test_loader.pt')
        print("done")

    optim = SGD(model.parameters(), lr=lr)
    # optim = AdamW(model.parameters(), lr=lr)
    # scheduler = ExponentialLR(optim, gamma=0.9)
    cost = nn.CrossEntropyLoss() if classification else nn.MSELoss()

    loss_stat = correct = total = 0
    start_time = time.time()
    i = 0

    for epoch in range(epochs):
        for x, y in train_loader:
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
                if paper:
                    x = x.unsqueeze(1).unsqueeze(1)

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
            i += 1
        # scheduler.step()

        correct = total = 0
        y_pred_all = None
        y_test_all = None

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

            if epoch == epochs - 1:
                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = torch.cat([y_pred_all, y_pred], dim=0)
                if y_test_all is None:
                    y_test_all = y
                else:
                    y_test_all = torch.cat([y_test_all, y], dim=0)

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print('epoch: {}, accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total, 100. * correct / total))
        print('time: {}'.format(time.time() - start_time))

        writer.add_scalar('Test/Loss', loss_stat / log_interval, (epoch + 1) * len(train_loader))
        writer.add_scalar('Test/Acc', 100. * correct / total, (epoch + 1) * len(train_loader))

    # y_test_all = torch.nn.functional.one_hot(y_test_all, num_classes=7)
    # y_pred_all = torch.nn.functional.one_hot(y_pred_all, num_classes=7)
    y_pred_all = torch.argmax(y_pred_all, -1)
    conf_matrix = confusion_matrix(y_true=y_test_all, y_pred=y_pred_all)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(f"{args.log_dir}/{args.name}/conf_matrix.png")

    writer.flush()
