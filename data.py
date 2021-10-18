import glob
import torch
from torch.utils.data import Dataset


class XRDData(Dataset):
    def __init__(self, root, train=True, train_test_split=0.9):
        self.xrds = []
        sub_dirs = glob.glob(root + "/*/")
        for y, class_name in enumerate(sub_dirs):
            class_data = []
            files = glob.glob(class_name + "/*")
            for file in files:
                with open(file) as f:
                    lines = f.readlines()[3:]
                    lines = [list(map(float, line.strip().split(", "))) for line in lines]
                    x = torch.FloatTensor(lines)
                    data_point = (x, y)
                    class_data.append(data_point)
            if train:
                class_data = class_data[:round(len(class_data) * train_test_split)]
            else:
                class_data = class_data[round(len(class_data) * train_test_split):]

            self.xrds = self.xrds + class_data

    def __len__(self):
        return len(self.xrds)

    def __getitem__(self, idx):
        return self.xrds[idx]
