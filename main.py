import torch
import numpy as np
import librosa as lb
from utils import *
from dataset.GTZANDataset import GTZANDataset

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

if __name__ == '__main__':
    cfg = read_yaml()
    main(cfg)