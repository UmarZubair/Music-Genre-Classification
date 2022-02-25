import torch
import numpy as np
from utils import *
from model.net import Net
from dataset.GTZANDataset import GTZANDataset
from serialize_data.serialize_data import serialize_data
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(cfg, device):
    print('\nStarting training!\n')
    train_df = load_csv(cfg['paths']['train_csv_path'])

    for fold in range(cfg['folds']['train_folds']):
        print('\nStarting Fold: {}'.format(fold + 1))
        df_val = train_df[train_df['fold'] == fold].drop(columns = ['fold'])
        df_train = train_df[train_df['fold'] != fold].drop(columns = ['fold'])

        train_dataset = GTZANDataset(df_train)
        val_dataset = GTZANDataset(df_val)
        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=True)
        
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=True)
        
        net = Net()
        net.to(device).train()
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr= float(cfg['train']['lr']))
        criterion = torch.nn.CrossEntropyLoss()
        lowest_val_loss = float(cfg['train']['lowest_val_loss'])
        
        for epoch in range(cfg['train']['epochs']):
            print('\nEpoch: {}\n'.format(epoch + 1))
            train_loss, train_accuracy = train_one_epoch(train_dataloader, optimizer, criterion, net, device)
            val_loss, val_accuracy= validate_one_epoch(val_dataloader, criterion, net, device)
            print('Train loss: {}\tTrain accuracy: {}\nValidation loss: {}\tValidation accuracy: {}'.format(train_loss, 
                                                                                                      train_accuracy, 
                                                                                                      val_loss,
                                                                                                      val_accuracy))
            # Check early stopping conditions.
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                patience_counter = 0
                print('Val loss improved. Saving weights')
                loss = np.round(np.array(val_loss),3)
                #best_path = cfg['paths']['weights_path'] + f"fold_{fold + 1}_{loss}_model.pth"
                best_path = cfg['paths']['weights_path'] + f"fold_{fold + 1}_model.pth"
                torch.save(net.state_dict(), best_path)
            else:
                patience_counter += 1
                if patience_counter > cfg['train']['patience']:
                    print('Early stopping. Exitting training. ')
                    break   
                
        # Evaluate the fold
        evaluate(cfg, best_path, device)


def train_one_epoch(data_loader, optimizer, criterion, net, device):
    running_loss = 0
    dataset_size = 0
    correct = 0
    
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
        predicted = torch.argmax(pred, 1)
        correct += predicted.eq(torch.argmax(target, 1)).sum().item()
        
    
    epoch_accuracy = 100 * correct / len(data_loader.dataset)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss, epoch_accuracy
        
def validate_one_epoch(data_loader, criterion, net, device):
    net.eval()
    running_loss = 0
    preds = []
    targets = []
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if device != 'cpu':
            data = data.to(device)
            target = target.to(device)
        
        batch_size = data.size(0)
        pred = net(data.unsqueeze(1))
        loss = criterion(pred, target)
        
        preds.append(pred.view(-1).cpu().detach().numpy())
        targets.append(target.view(-1).cpu().detach().numpy())
        running_loss += loss.item() * batch_size
        predicted = torch.argmax(pred, 1)
        correct += predicted.eq(torch.argmax(target, 1)).sum().item()
    
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    epoch_accuracy = np.round(np.array(100 * correct / len(data_loader.dataset)),3)   
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss, epoch_accuracy
            
        
def evaluate(cfg, weights_path, device):
    print('\nStarting evaluating!\n')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(weights_path))
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    test_df = load_csv(cfg['paths']['test_csv_path'])
    test_dataset = GTZANDataset(test_df)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=cfg['train']['batch_size'], 
                                    shuffle=False)
    running_loss = 0
    preds = []
    targets = []
    correct = 0
    with torch.no_grad(): 
        for batch_idx, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)
            
            batch_size = data.size(0)
            pred = net(data.unsqueeze(1))
            loss = criterion(pred, target)
            
            preds.append(pred.view(-1).cpu().detach().numpy())
            targets.append(target.view(-1).cpu().detach().numpy())
            running_loss += loss.item() * batch_size
            predicted = torch.argmax(pred, 1)
            correct += predicted.eq(torch.argmax(target, 1)).sum().item()

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    epoch_accuracy = np.round(np.array(100 * correct / len(test_dataloader.dataset)),3)   
    epoch_loss = running_loss / len(test_dataloader.dataset)
    print('\nTest accuracy: {}\t Test loss: {}'.format(epoch_accuracy, epoch_loss))

    
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
    device = cfg['defaults']['device']
    print('\nDevice on: {}'.format(device))
    
    if cfg['defaults']['serialize_data']:
        clear_feature_folders(cfg)
        serialize_data(cfg)
        
    if cfg['defaults']['train']:
        train(cfg, device)
    
    if cfg['defaults']['evaluate']:
        evaluate(cfg, cfg['paths']['best_weight_path'], device)
    
    #testing()  
    #test_2(cfg)
    #main(cfg)   
    #testing()