import torch
import numpy as np
import librosa as lb
from utils import *
from dataset.GTZANDataset import GTZANDataset
from serialize_data.serialize_data import serialize_data
from pickle import load as pickle_load
from path import Path

def main(cfg):
    df = load_csv(cfg['paths']['csv_path'])
    print(df)
    #sound = lb.load(df.iloc[0][0])
    #print(sound)
    #print(df.iloc[0][0])
    for fold in range(5):
        test_df = df[df['fold'] == fold]
        train_df = df[df['fold'] != fold]
    
    #test_df = df[df['fold'] == 0]
    #print(test_df)
    #train_df = df[df['fold'] != 0]
    #print(train_df)

    #training_dataset = GTZANDataset(train_df)
    #testing_dataset = GTZANDataset(test_df)

def test(cfg):
    df = load_csv(cfg['paths']['train_csv_path'])
    file_path = Path(df.iloc[0]['feature_path'])
    with file_path.open('rb') as f:
            dict = pickle_load(f)
    x = dict['features']
    y = dict['class'].split(',')
    print(np.shape(x))
    print(np.shape(y))
    print(y)
    
    
if __name__ == '__main__':
    cfg = read_yaml()
    
    if cfg['defaults']['serialize_data']:
        serialize_data(cfg)
        
    test(cfg)
    #main(cfg)