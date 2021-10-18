import glob
import torch
from torch.utils.data import Dataset


class XRDData(Dataset):
    def __init__(self, root, train=True):
        self.xrds = []
        sub_dirs = glob.glob(root + "/*/")
        print(sub_dirs)
        for class_ind, class_name in enumerate(sub_dirs):
            class_data = []
            files = glob.glob(class_name)
            for file in files:
                with open(file) as file:
                    lines = file.readlines()[3:]
                    x = torch.FloatTensor([line.strip().split() for line in lines])
                    data_point = (x, class_ind)
                    class_data.append(data_point)
            if train:
                class_data = class_data[:round(len(class_data) * 0.9)]
            else:
                class_data = class_data[round(len(class_data) * 0.9):]

            self.xrds = self.xrds + class_data

    def __len__(self):
        return len(self.xrds)

    def __getitem__(self, idx):
        return self.xrds[idx]
