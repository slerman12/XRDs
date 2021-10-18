import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class XRDData(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        self.xrds = []
        sub_dirs = glob.glob(root + "/*/")
        for y, class_name in enumerate(sub_dirs):
            class_data = []
            files = glob.glob(class_name + "/*")
            # todo its fast to just count the sizes... can count sizes and get item via just reading one at a time
            for file in tqdm(files, desc="Reading " + class_name):
                with open(file) as f:
                    lines = f.readlines()[3:]
                    lines = lines[:round(len(class_data) * train_test_split)] if train \
                        else lines[round(len(class_data) * train_test_split):]
                    lines = [list(map(float, line.strip().split(", "))) for line in lines]
                    x = torch.FloatTensor(lines)
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
                    self.xrds[class_name] = lines, len(self.xrds[class_name])
        self.train = train
        self.train_test_split = train_test_split

    def __len__(self):
        size = 0
        for class_name in self.xrds:
            for lines in self.xrds:
                total_size = len(lines)
                train_size = round(total_size * self.train_test_split)
                if self.train:
                    size += train_size
                else:
                    size = total_size - train_size
        return size

    def __getitem__(self, idx):
        for class_name in self.xrds:
            for y, lines in enumerate(self.xrds):
                lines = [list(map(float, line.strip().split(", "))) for line in lines]
                x = torch.FloatTensor(lines)
                return x, y
