import torch
import torch.nn as nn
import torch.nn.functional as F 



class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()


    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss
     