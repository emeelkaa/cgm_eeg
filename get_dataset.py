import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import resample

class EEGDataset(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), 'rb'))
        X = sample['X']
        Y = torch.LongTensor(sample['y'])
        X = torch.FloatTensor(X)
        return X, Y

class CHBMITDataset(EEGDataset):
    def __init__(self, root, files, sfreq):
        super().__init__(root, files)
        self.default_rate = 200
        self.sfreq = sfreq

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sfreq != self.default_rate:
            X = resample(X, 10 * self.sfreq, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = torch.tensor(sample['y'], dtype=torch.long)
        X = torch.FloatTensor(X)
        return X, Y

class TUEVDataset(EEGDataset):
    def __init__(self, root, files, sfreq):
        super().__init__(root, files)
        self.default_rate = 250 
        self.sfreq = sfreq
    
    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sfreq != self.default_rate:
            X = resample(X, 5 * self.sfreq, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = torch.tensor(int(sample["label"][0]) - 1, dtype=torch.long)
        X = torch.FloatTensor(X)
        return X, Y

def get_dataset(args, verbose=False):
    if args.dataset.lower() == 'chbmit':
        args.n_channels = 16 
        args.sfreq = 200
        args.n_classes = 1
        root = "../../data/chbmit/clean_segments"
        train_files = sorted(os.listdir(os.path.join(root, "train")))  
        val_files = sorted(os.listdir(os.path.join(root, "val")))
        test_files = sorted(os.listdir(os.path.join(root, "test")))

        train_dataset = CHBMITDataset(os.path.join(root, "train"), train_files, 200)
        val_dataset = CHBMITDataset(os.path.join(root, "val"), val_files, 200)
        test_dataset = CHBMITDataset(os.path.join(root, "test"), test_files, 200)

    elif args.dataset.lower() == 'tuev':
        args.n_channels = 16 
        args.sfreq = 250
        args.n_classes = 6
        root = "../../data/tuev/edf"
        files = sorted(os.listdir(os.path.join(root, "processed_train")))  

        train_sub = sorted(set([f.split("_")[0] for f in files]))          
        val_sub = set(np.random.choice(train_sub, size=int(len(train_sub) * 0.2), replace=False))
        train_sub = sorted(set(train_sub) - val_sub)                       

        train_files = [f for f in files if f.split("_")[0] in train_sub]
        val_files = [f for f in files if f.split("_")[0] in val_sub]       
        test_files = sorted(os.listdir(os.path.join(root, "processed_eval")))

        train_dataset = TUEVDataset(os.path.join(root, "processed_train"), train_files, 250)
        val_dataset = TUEVDataset(os.path.join(root, "processed_train"), val_files, 250)
        test_dataset = TUEVDataset(os.path.join(root, "processed_eval"), test_files, 250)
        
    else: 
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if verbose: 
        print("Datasets successfully created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
    return train_dataset, val_dataset, test_dataset, args