from utils import *
from feature_extract import extract_mel_band_energies
import os

def prepare_datasets():
    cfg = read_yaml()
    df = create_csv(cfg['class_names'], cfg['paths']['data_path'])
    
    # Create 10 folds
    df = create_folds(df, 10)
    
    # 1 fold for testing
    test_df = df[df['fold'] == 9].drop(columns=['fold'])
    
    # 9 folds for training
    train_df = df[df['fold'] != 9].drop(columns=['fold']).reset_index(drop = True)
    train_df = create_folds(train_df, 5)

    # Add target column to dataframe as one hot encoding
    test_df = add_targets_to_df(test_df)
    train_df = add_targets_to_df(train_df)

    # Save the files
    test_df.to_csv('data/test_metadata.csv', index=False)
    train_df.to_csv('data/train_metadata.csv', index=False)


def main():
    modes = ['train', 'test']
    print('Preparing datasets...')
    
    for mode in modes:
        df = load_csv('data/' + mode + '_metadata.csv')
        features_and_classes = {}
        df['feature_path'] = -1
        
        for i in range(len(df)):
            audio, _ = get_audio_file_data(df.iloc[i]['audio_path'])
            file_name = df.iloc[i]['audio_path'].split('\\')[-1].replace('.au', '.pickle')
            MBE = extract_mel_band_energies(audio)
            features_and_classes.update({'features': MBE, 'class': df.iloc[i]['target']})

            if not os.path.exists('data/' + mode + '_features'):
                os.mkdir('data/' + mode + '_features')
                
            feature_path = 'data/' + mode + '_features/' + file_name
            df['feature_path'][i] = feature_path
            serialize_features_and_classes(features_and_classes, feature_path)
        
        # Update csv and save
        df = df.drop(columns=['target'])
        df = df.to_csv('data/' + mode + '_metadata.csv', index=False)
    print('Done!')
    
if __name__ == "__main__":
    main()