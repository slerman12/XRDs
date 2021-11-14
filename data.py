import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class XRDData(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        self.num_datapoints = 43049
        self.train_test_split = train_test_split
        self.size = train_size = round(self.num_datapoints * self.train_test_split)
        self.train = train
        if not self.train:
            self.size = self.num_datapoints - train_size

        self.feature_file = root + "/43049_features.csv"
        self.label_file = root + "/43049_labels.csv"

        self.train_inds = np.random.choice(np.arange(self.num_datapoints), size=train_size, replace=False)
        self.test_inds = np.array([x for x in np.arange(self.num_datapoints) if x not in self.train_inds])

        with open(self.feature_file) as f:
            self.feature_lines = f.readlines()
        with open(self.label_file) as f:
            self.label_lines = f.readlines()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.train:
            idx = self.train_inds[idx]
        else:
            idx = self.test_inds[idx]
        # if not self.train:
        #     idx = idx + round(self.num_datapoints * self.train_test_split)
        line = self.feature_lines[idx]
        x = list(map(float, line.strip().split(", ")))
        x = torch.FloatTensor(x)
        line = self.label_lines[idx]
        y = list(map(float, line.strip().split(", ")))
        y = torch.FloatTensor(y)
        y = torch.argmax(y)
        return x, y


class XRDDataOld(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        self.xrds = []
        sub_dirs = glob.glob(root + "/*/")
        for class_name in sub_dirs:
            files = glob.glob(class_name + "/*")
            files = files[:round(len(files) * train_test_split)] if train \
                else files[round(len(files) * train_test_split):]
            for file in tqdm(files, desc="Reading " + class_name):
                with open(file) as f:
                    lines = f.readlines()
                    y = int(lines[1][-1])
                    lines = lines[3:]
                    lines = [list(map(float, line.strip().split(", "))) for line in lines]
                    x = torch.FloatTensor(lines)
                    if torch.isnan(x).any():
                        print("File", file, "in", class_name, "contains NaN")
                    data_point = (x, y)
                    self.xrds.append(data_point)

    def __len__(self):
        return len(self.xrds)

    def __getitem__(self, idx):
        return self.xrds[idx]


class XRDDataFast(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        sub_dirs = glob.glob(root + "/*/")
        self.xrds = {}
        self.size = 0
        for class_name in sub_dirs:
            files = glob.glob(class_name + "/*")
            for file in files:
                with open(file) as f:
                    lines = f.readlines()[3:]
                    self.xrds[class_name] = lines
                    total_size = len(lines)
                    train_size = round(total_size * train_test_split)
                    if train:
                        self.size += train_size
                    else:
                        self.size = total_size - train_size
        self.train = train
        self.train_test_split = train_test_split

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # todo index and train/test split
        for class_name in self.xrds:
            for y, lines in enumerate(self.xrds):
                lines = [list(map(float, line.strip().split(", "))) for line in lines]
                x = torch.FloatTensor(lines)
                return x, y
