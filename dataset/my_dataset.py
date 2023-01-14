from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Excel_dataset(Dataset):

    def __init__(self, dir, if_normalize=False, if_onehot=False):
        super(Excel_dataset, self).__init__()

        if (dir.endswith('.csv')):
            data = pd.read_csv(dir)
        elif (dir.endswith('.xlsx') or dir.endswith('.xls')):
            data = pd.read_excel(dir)

        nplist = data.T.to_numpy()
        data = nplist[0:-1].T
        self.data = np.float64(data)
        self.target = nplist[-1]

        self.target_type = []

        #  trans to tensor
        self.data = np.array(self.data)
        self.data = torch.FloatTensor(self.data)
        self.target = np.array(self.target)
        self.target = torch.FloatTensor(self.target)
        self.if_onehot = if_onehot

        if if_normalize == True:
            self.data = nn.functional.normalize(self.data)

    def __getitem__(self, index):
        if self.if_onehot == True:
            return self.data[index], self.target_onehot[index]

        else:
            return self.data[index].unsqueeze(0), self.target[index]

    def __len__(self):
        return len(self.target)


class Excel_dataset_test(Dataset):

    def __init__(self, dir, if_normalize=False, if_onehot=False):
        super(Excel_dataset_test, self).__init__()

        if (dir.endswith('.csv')):
            data = pd.read_csv(dir)
        elif (dir.endswith('.xlsx') or dir.endswith('.xls')):
            data = pd.read_excel(dir)

        nplist = data.T.to_numpy()
        data = nplist[0:4].T
        self.data = np.float64(data)

        # trans to tensor
        self.data = np.array(self.data)
        self.data = torch.FloatTensor(self.data)
        self.if_onehot = if_onehot

        if if_normalize == True:
            self.data = nn.functional.normalize(self.data)

    def __getitem__(self, index):
        if self.if_onehot == True:
            return self.data[index]

        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def data_split(data, rate):
    train_l = int(len(data) * rate)
    test_l = len(data) - train_l
    # Shuffle the dataset and split it
    train_set, test_set = torch.utils.data.random_split(data, [train_l, test_l])
    return train_set, test_set

