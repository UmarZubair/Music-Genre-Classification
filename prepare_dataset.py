from utils import *

def prepare_datasets():
    cfg = read_yaml()
    df = create_csv(cfg['class_names'], cfg['paths']['data_path'])
    
    # Create 10 folds
    df = create_folds(df, 10)
    
    # 1 fold for testing
    test_data_df = df[df['fold'] == 9].drop(columns=['fold'])
    
    # 9 folds for training
    train_data_df = df[df['fold'] != 9].drop(columns=['fold']).reset_index(drop = True)
    train_data_df = create_folds(train_data_df, 5)

    # Add target column to dataframe as one hot encoding
    test_data_df = add_targets_to_df(test_data_df)
    train_data_df = add_targets_to_df(train_data_df)

    # Save the files
    test_data_df.to_csv('data/test_metadata.csv', index=False)
    train_data_df.to_csv('data/train_metadata.csv', index=False)

if __name__ == "__main__":
    prepare_datasets()