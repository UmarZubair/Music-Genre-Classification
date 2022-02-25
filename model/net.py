import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

cfg = utils.read_yaml()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=cfg['model']['conv1_in_channel'],
                               out_channels=cfg['model']['conv1_out_channel'],
                               kernel_size=cfg['model']['conv1_kernel_size'])
        self.conv2 = nn.Conv2d(in_channels=cfg['model']['conv2_in_channel'],
                               out_channels=cfg['model']['conv2_out_channel'],
                               kernel_size=cfg['model']['conv2_kernel_size'],
                               stride=cfg['model']['conv2_stride'])   
        
        self.dropout1 = nn.Dropout(cfg['model']['dropout_1'])
        self.dropout2 = nn.Dropout(cfg['model']['dropout_2'])
        
        self.fc1 = nn.Linear(in_features=cfg['model']['fc1_in_features'],
                             out_features=cfg['model']['fc1_out_features'])     
        self.fc2 = nn.Linear(in_features=cfg['model']['fc2_in_features'],
                             out_features=cfg['model']['fc2_out_features'])
    
    def forward(self, features):
        x = self.conv1(features)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, cfg['model']['max_pool'])
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        out = self.fc2(x)
        return out
        