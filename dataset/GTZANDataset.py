from pickle import load as pickle_load
from pathlib import Path
from sklearn.datasets import load_files
import torch
import numpy as np

class GTZANDataset:
    def __init__(self, df):
        self.paths = df['feature_path']

    def load_file(self,file_path):
        with file_path.open('rb') as f:
            dict = pickle_load(f)
        return dict['features'], dict['class']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
       path = self.paths[idx]
       features, target = self.load_file(Path(path))
       target = np.array(target.split(',')).astype('float')
       return torch.tensor(features), torch.tensor(target)