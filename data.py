import glob
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
import numpy as np


class XRDData(Dataset):
    def __init__(self, root, num_classes=230, deliminator=','):
        self.feature_file = root + "/features.csv"
        self.label_file = root + f"/labels{num_classes}.csv"

        with open(self.feature_file) as f:
            self.feature_lines = f.readlines()
        with open(self.label_file) as f:
            self.label_lines = f.readlines()

        self.size = len(self.feature_lines)
        assert self.size == len(self.label_lines), 'num features and labels not same'

        labels = torch.stack([torch.argmax(torch.FloatTensor(list(map(float, line.strip().split(deliminator))))) for i, line in enumerate(self.label_lines)])
        self.y_count = {c: (labels == c).sum() for c in range(num_classes)}

        self.deliminator = deliminator

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        line = self.feature_lines[idx]
        x = list(map(float, line.strip().split(self.deliminator)))
        x = torch.FloatTensor(x)

        line = self.label_lines[idx]
        y = list(map(float, line.strip().split(self.deliminator)))
        y = torch.FloatTensor(y)
        y = torch.argmax(y)
        return x, y
