from re import T
import torch
import numpy as np
import librosa as lb
from utils import *
from dataset.GTZANDataset import GTZANDataset
from serialize_data.serialize_data import serialize_data,prepare_datasets
from pickle import load as pickle_load
#from path import Path
import glob

def main(cfg):
    train_df = load_csv(cfg['paths']['train_csv_path'])
    test_df = load_csv(cfg['paths']['test_csv_path'])

    for fold in range(5):
        print(f'Starting Fold: {fold + 1}')
        df_val = train_df[train_df['fold'] == fold].drop(columns = ['fold'])
        df_train = train_df[train_df['fold'] != fold].drop(columns = ['fold'])


        train_dataset = GTZANDataset(df_train)
        val_dataset = GTZANDataset(df_val)

        #dataloaders

        #accuracy for validation and training accuracy

        #accuracy for testing

        

    
if __name__ == '__main__':
    cfg = read_yaml()
    if cfg['defaults']['serialize_data']:
        serialize_data(cfg)

    #test(cfg)  
    #test_2(cfg)
    main(cfg)   