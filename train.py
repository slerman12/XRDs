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

# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', default="runs", type=str, help='logging directory')
parser.add_argument('--name', default="dnn", type=str, help='logging directory')
parser.add_argument('--num-workers', default=0, type=int, help='number data loading workers')
parser.add_argument('--num-classes', default=7, type=int, help='number classes')
args = parser.parse_args()


seed = 1
torch.manual_seed(seed)
random.seed(seed)

classification = True
conv = False
paper = False
resize = False
num_classes = args.num_classes
# root = train_root = test_root = 'data_01_23'
# deliminator = ', '
subspace = None
train_root = 'domain_adaptation_data/domainAdaptation/synthetic_domain'
test_root = 'domain_adaptation_data/domainAdaptation/rruff_domain'
train_num_datapoints = 171009
test_num_datapoints = 549
train_split = 1
test_split = 0
deliminator = ','
# subspace = [50, 900]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNet1DPaper(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet1DPaper, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=80, stride=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),
            nn.Conv1d(80, 80, kernel_size=50, stride=5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Conv1d(80, 80, kernel_size=25, stride=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(160, 140),
            nn.ReLU(),
            nn.Linear(140, 70),
            nn.ReLU(),
            nn.Linear(70, num_classes))

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out


class ConvNet1D(nn.Module):
    def __init__(self, num_classes=num_classes):
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

#
# class ConvNet1DResize(nn.Module):
#     def __init__(self, num_classes=num_classes):
#         super(ConvNet1DResize, self).__init__()
#         self.CNN = nn.Sequential(
#             nn.Conv1d(1, 256, kernel_size=3, stride=1, padding=1),  # Conserves height/width
#             nn.BatchNorm1d(256),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),  # Conserves height/width
#             nn.BatchNorm1d(256),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),  # Conserves height/width
#             nn.BatchNorm1d(256),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             nn.MaxPool1d(kernel_size=2, stride=2),  # Cuts height/width in 2
#             nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),  # Conserves height/width
#             nn.BatchNorm1d(256),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             nn.Flatten(),
#             nn.Linear(425 * 256, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes),
#         )
#
#     def forward(self, x):
#         return self.CNN(x)


# class ConvNet1DResize(nn.Module):
#     def __init__(self, num_classes=num_classes):
#         super(ConvNet1DResize, self).__init__()
#         self.CNN = \
#             nn.Sequential(
#                 nn.Conv1d(1, 80, (100,), (5,)),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.AvgPool1d(3, 2),
#                 nn.Conv1d(1, 80, (50,), (5,)),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.AvgPool1d(3),
#                 nn.Conv1d(1, 80, (25,), (2,)),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.AvgPool1d(3),
#                 nn.Flatten(),
#                 nn.Linear(features, 2300),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(2300, 1150),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(512, num_classes)
#             )
#
#     def forward(self, x):
#         return self.CNN(x)


# class ConvNet1DResize(nn.Module):
#     def __init__(self, num_classes=num_classes):
#         super(ConvNet1DResize, self).__init__()
#         self.CNN = nn.Sequential(
#             # Reduce kernel size  5 -> 3
#             # Reduce padding  2 -> 1
#             # Increase channel width  16 -> 64
#             nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),  # Conserves height/width
#             nn.BatchNorm1d(64),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             nn.MaxPool1d(kernel_size=2, stride=2),  # Cuts height/width in 2
#             # Increase channel width  32 -> 128
#             nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),  # Conserves height/width
#             nn.BatchNorm1d(128),  # Conserves height/width
#             nn.ReLU(),  # Conserves height/width
#             # Removed MaxPool
#             nn.Flatten(),
#             nn.Linear(425 * 128, num_classes))
#
#     def forward(self, x):
#         return self.CNN(x)


class ConvNet1DResize(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet1DResize, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=10, stride=2),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv1d(256, 256, kernel_size=(10,), stride=(2,)),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv1d(256, 256, kernel_size=10, stride=2),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv1d(256, 256, kernel_size=(10,), stride=(2,)),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.CNN(x)


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


if "dnn" == args.name:
    model = nn.Sequential(nn.Linear(1800, 512), nn.ReLU(),
                          nn.Linear(512, 256), nn.ReLU(),
                          nn.Linear(256, 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(),
                          nn.Linear(64, num_classes))


elif "dnn_resize" == args.name:
    model = nn.Sequential(nn.Linear(850, 8500), nn.ReLU(),
                          nn.Linear(8500, 8500), nn.ReLU(),
                          nn.Linear(8500, num_classes))

    # model = nn.Sequential(nn.Linear(850, 256), nn.ReLU(),
    #                       nn.Linear(256, 128), nn.ReLU(),
    #                       nn.Linear(128, 64), nn.ReLU(),
    #                       nn.Linear(64, num_classes))
elif args.name == "cnn":
    model = ConvNet1D()
    conv = True
elif args.name == "cnnp":
    model = ConvNet1DPaper()
    conv = True
    paper = True
elif args.name == "cnn_resize":
    model = ConvNet1DResize()
    conv = True
    resize = True
elif "logreg" == args.name:
    model = nn.Sequential(nn.Linear(1800, num_classes))
elif "logreg_resize" == args.name:
    model = nn.Sequential(nn.Linear(850, num_classes))
else:
    assert False

model = model.to(device)

writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.name}")

if conv:
    if paper:
        summary(model, (1, 1800))
    elif resize:
        summary(model, (1, 850))
    else:
        summary(model, (1, 1800))


if __name__ == '__main__':
    epochs = 2000
    log_interval = 1000
    batch_size = 32
    lr = 0.01

    train_test_split = 0.9
    print("parsing train...")
    train_dataset = XRDData(train_root, train=True, train_test_split=train_split, num_classes=num_classes,
                            deliminator=deliminator, subspace=subspace, num_datapoints=train_num_datapoints)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    print("done")

    print("parsing test...")
    test_dataset = XRDData(test_root, train=False, train_test_split=test_split, num_classes=num_classes,
                           deliminator=deliminator, num_datapoints=test_num_datapoints)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("done")

    # optim = SGD(model.parameters(), lr=lr)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # scheduler = ExponentialLR(optim, gamma=0.9)
    # cost = nn.CrossEntropyLoss(reduction='none') if classification else nn.MSELoss()
    cost = nn.CrossEntropyLoss() if classification else nn.MSELoss()

    loss_stat = correct = total = 0
    start_time = time.time()
    i = 0

    # def balance(labels):
    #     return 1 / torch.tensor([train_dataset.y_count[int(l)] for l in labels]).to(device)

    for epoch in range(epochs):

        model = model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

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
                # assert x.shape[2] == 1800

            # one_hot = F.one_hot(y, num_classes=10).float()
            y_pred = model(x)
            # loss = (cost(y_pred, y) * balance(y)).mean()
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

        model = model.eval()

        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

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
                # assert x.shape[2] == 1800
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

    y_pred_all = torch.argmax(y_pred_all, -1)

    for label in range(num_classes):
        if len(y_test_all[y_test_all == label]) > 0:
            acc = y_pred_all[y_test_all == label]
            acc = len(acc[acc == label]) / max(len(acc), 1)
            print(f"Accuracy for class {label}: {acc}")

    # y_test_all = torch.nn.functional.one_hot(y_test_all, num_classes=7)
    # y_pred_all = torch.nn.functional.one_hot(y_pred_all, num_classes=7)

    # conf_matrix = confusion_matrix(y_true=y_test_all, y_pred=y_pred_all)

    # fig, ax = plt.subplots(figsize=(num_classes, num_classes))
    # ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.savefig(f"{args.log_dir}/{args.name}/conf_matrix.png")

    writer.flush()
