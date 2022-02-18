from utils import *
from serialize_data.feature_extract import extract_mel_band_energies
import os

def prepare_datasets(cfg):
    """ Create csv files for training and testing datasets with folds 
    
    :param cfg: Configuration dictionary
    """
    df = create_csv(cfg['class_names'], cfg['paths']['audio_path'])
    
    # Create 10 folds
    df = create_folds(df, cfg['folds']['num_folds'])
    
    # 1 fold for testing
    test_df = df[df['fold'] ==  cfg['folds']['fold_for_testing']].drop(columns=['fold'])
    
    # 9 folds for training
    train_df = df[df['fold'] != cfg['folds']['fold_for_testing']].drop(
                columns=['fold']).reset_index(drop = True)
    train_df = create_folds(train_df, cfg['folds']['train_folds'])

    # Add target column to dataframe as one hot encoding
    test_df = add_targets_to_df(test_df)
    train_df = add_targets_to_df(train_df)

    # Save the files
    test_df.to_csv(cfg['paths']['test_csv_path'], index=False)
    train_df.to_csv(cfg['paths']['train_csv_path'], index=False)

def serialize_data(cfg):
    """ Serialize the data into pickle files 
    
    :param cfg: Configuration dictionary
    """
    prepare_datasets(cfg)
    modes = ['train', 'test']
    print('Serializing datasets...')
    
    for mode in modes:
        df = load_csv(cfg['paths']['root_path'] + mode + '_metadata.csv')
        features_and_classes = {}
        df['feature_path'] = -1
        
        for i in range(len(df)):
            audio, _ = get_audio_file_data(df.iloc[i]['audio_path'])
            file_name = df.iloc[i]['audio_path'].split('\\')[-1].replace('.au', '.pickle')
            MBE = extract_mel_band_energies(audio)
            features_and_classes.update({'features': MBE, 'class': df.iloc[i]['target']})

            if not os.path.exists(cfg['paths']['root_path'] + mode + '_features'):
                os.mkdir(cfg['paths']['root_path'] + mode + '_features')
                
            feature_path = cfg['paths']['root_path'] + mode + '_features/' + file_name
            df['feature_path'][i] = feature_path
            serialize_features_and_classes(features_and_classes, feature_path)
        
        # Update csv and save
        df = df.drop(columns=['target'])
        df = df.to_csv(cfg['paths']['root_path'] + mode + '_metadata.csv', index=False)
        
    print('Done!')
    