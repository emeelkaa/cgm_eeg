import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class CHBDataset(Dataset):
    def __init__(self, root: str, files: list):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), 'rb'))
        X = sample['X']
        Y = sample['y']
        X = torch.FloatTensor(X)
        return X, Y

class TUEVDataset(Dataset):
    def __init__(self, root: str, files: list):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), 'rb'))
        X = sample['X']
        Y = sample['y']
        X = torch.FloatTensor(X) 
        Y = Y - 1
        return X, Y


class TUSZDataset(Dataset):
    def __init__(self, root: str, files: list):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), 'rb'))
        X = sample['X']
        Y = sample['y']
        X = torch.FloatTensor(X) 
        if Y == 4:
            Y = 3
        return X, Y


def get_chbmit(subject_independent=False):
    if subject_independent:
        root = "chbmit/clean_segments"
    else:
        root = "chbmit/clean_segments_2"

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    train_dataset = CHBDataset(os.path.join(root, "train"), train_files)
    val_dataset = CHBDataset(os.path.join(root, "val"), val_files)
    test_dataset = CHBDataset(os.path.join(root, "test"), test_files)

    return train_dataset, val_dataset, test_dataset


def get_tuev():
    root = "tuev"
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    train_dataset = TUEVDataset(os.path.join(root, "train"), train_files)
    val_dataset = TUEVDataset(os.path.join(root, "val"), val_files)
    test_dataset = TUEVDataset(os.path.join(root, "test"), test_files)
    return train_dataset, val_dataset, test_dataset

def get_tusz():
    root = "tusz"
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    train_dataset = TUSZDataset(os.path.join(root, "train"), train_files)
    val_dataset = TUSZDataset(os.path.join(root, "val"), val_files)
    test_dataset = TUSZDataset(os.path.join(root, "test"), test_files)
    return train_dataset, val_dataset, test_dataset