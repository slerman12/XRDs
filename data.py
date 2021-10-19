import glob
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class XRDData(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        self.xrds = []
        sub_dirs = glob.glob(root + "/*/")
        for y, class_name in enumerate(sub_dirs):
            files = glob.glob(class_name + "/*")
            files = files[:round(len(files) * train_test_split)] if train \
                else files[round(len(files) * train_test_split):]
            for file in tqdm(files, desc="Reading " + class_name):
                with open(file) as f:
                    lines = f.readlines()[3:]
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
