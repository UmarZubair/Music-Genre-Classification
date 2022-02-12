import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
import glob

def create_folds(csv_save_path):
    cfg = read_yaml()
    df = create_csv(cfg['class_names'], cfg['paths']['data_path'])
    df['fold'] = -1
    target = df['target'].to_numpy()
    skf = StratifiedKFold(n_splits=cfg['num_of_folds'], shuffle=True)
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(df,target)):
        df.loc[val_idx,'fold'] = fold_num
    print(df)
    df.to_csv(csv_save_path, index=False)

def create_csv(class_names, data_path):
    col_names = ['audio_path','target']
    df = pd.DataFrame(columns=col_names)
    for name in class_names:
        paths = glob.glob(data_path + name + '/*.au')
        for path in paths:
            df = df.append({col_names[0]: path, col_names[1]:name}, ignore_index=True)
    return df

def read_yaml(file_path = 'config/config.yml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_csv(path):
    return pd.read_csv(path)

#cfg = read_yaml()
#create_folds(cfg['paths']['csv_path'])