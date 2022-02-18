import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from typing import  Union,MutableMapping
from pickle import dump
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import glob
from librosa.core import load as lb_load
from pathlib import Path

def create_folds(df, num_of_folds):
    cfg = read_yaml()
    df['fold'] = -1
    target = df['class_name'].to_numpy()
    skf = StratifiedKFold(n_splits=num_of_folds, shuffle=True)
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(df,target)):
        df.loc[val_idx,'fold'] = fold_num
    return df
    

def create_csv(class_names, data_path):
    col_names = ['audio_path','class_name']
    df = pd.DataFrame(columns=col_names)
    for name in class_names:
        paths = glob.glob(data_path + name + '\*.au')
        for path in paths:
            df = df.append({col_names[0]: Path(path), col_names[1]:name}, ignore_index=True)
    return df


def read_yaml(file_path = 'config/config.yml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_csv(path):
    return pd.read_csv(path)


def one_hot_encode(df):
    """Perform one-hot encoding on 'label' column 

    :param df: Dataframe 
    :returns data in one encode form
    """
    encoder = OneHotEncoder(handle_unknown='ignore')
    return encoder.fit_transform(df[['class_name']]).toarray()


def serialize_features_and_classes(features_and_classes: MutableMapping[str, Union[np.ndarray, int]], f_name: str):
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    try:
       pickle_out = open(f_name,"wb")
       dump(features_and_classes, pickle_out)
       pickle_out.close()
    except EOFError as e:
        print(e)


def get_audio_file_data(audio_file: str) \
        -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray   
    """
    return lb_load(path=audio_file, sr=None, mono=True)

def add_targets_to_df(df):
    df['target'] = one_hot_encode(df).tolist()
    df['target'] = [','.join(map(str, l)) for l in df['target']]
    return df
