import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class classifier(nn.Module):
    def __init__(self,n_classes,dropout):
        super(classifier, self).__init__()
        self.cl = nn.Linear(200,n_classes)
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.xavier_uniform_(self.cl.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self,x):
        x = self.cl(x)
        x = self.dropout(x)
        x = F.elu(x)
        return x
