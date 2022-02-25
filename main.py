from pickletools import optimize
import torch
import numpy as np
from utils import *
from model.net import Net
from dataset.GTZANDataset import GTZANDataset
from serialize_data.serialize_data import serialize_data,prepare_datasets
from pickle import load as pickle_load
from torch.utils.data import DataLoader
from serialize_data.feature_extract import extract_mel_band_energies
from tqdm import tqdm

def main(cfg):
    train_df = load_csv(cfg['paths']['train_csv_path'])
    test_df = load_csv(cfg['paths']['test_csv_path'])
    test_dataset = GTZANDataset(test_df)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=True)
        

    for fold in range(1):
        print(f'Starting Fold: {fold + 1}')
        df_val = train_df[train_df['fold'] == fold].drop(columns = ['fold'])
        df_train = train_df[train_df['fold'] != fold].drop(columns = ['fold'])

        train_dataset = GTZANDataset(df_train)
        val_dataset = GTZANDataset(df_val)
        
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=True)
        
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=False)
        
        device = cfg['defaults']['device']
        net = Net()
        net.to(device).train()
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr= cfg['train']['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(cfg['train']['epochs']):
            print('Epoch: {}\n'.format(epoch + 1))
            train_loss = train_one_epoch(train_dataloader, optimizer, criterion, net, device)
            val_loss = validate_one_epoch(val_dataloader, criterion, net, device)
            print('Acc Loss: {}\tVar Loss: {}'.format(train_loss, val_loss))
            

def train_one_epoch(data_loader, optimizer, criterion, net, device):
    running_loss = 0
    dataset_size = 0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if device != 'cpu':
            data = data.to(device)
            target = target.to(device)
        
        batch_size = data.size(0)
        optimizer.zero_grad()
        pred = net(data.unsqueeze(1))
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss
        
def validate_one_epoch(data_loader, criterion, net, device):
    net.eval()
    running_loss = 0
    preds = []
    targets = []
    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
        if device != 'cpu':
            data = data.to(device)
            target = target.to(device)
        
        batch_size = data.size(0)
        pred = net(data.unsqueeze(1))
        loss = criterion(pred, target)
        
        preds.append(pred.view(-1).cpu().detach().numpy())
        targets.append(target.view(-1).cpu().detach().numpy())
        running_loss += loss.item() * batch_size
        
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss
            
        
def testing():
    #audio,sr = get_audio_file_data('data/audio/classical/classical.00042.au')
    #spec = extract_mel_band_energies(audio)
    #plt.imshow(spec)
    #plt.plot()
    #file_path = Path('data/train_features/classical.00042.pickle')
    #with file_path.open('rb') as f:
    #        dict = pickle_load(f)
    #print(np.shape(dict['features']))
    #1print(np.shape(spec))
    #ld.specshow(spec, x_axis='time', y_axis='mel')
    net = Net()
    print(net)
    #plt.show()

    
if __name__ == '__main__':
    cfg = read_yaml()
         
    if cfg['defaults']['serialize_data']:
        clear_feature_folders(cfg)
        serialize_data(cfg)
    main( cfg)
    
    #testing()  
    #test_2(cfg)
    #main(cfg)   
    #testing()