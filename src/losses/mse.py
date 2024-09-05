import torch
import torch.nn as nn
import torch.nn.functional as F 



class Loss(nn.Module) :
    def __init__(self, reduction ='sum') -> None :
    
        super().__init__()
        self.reduction = reduction
        self.rc_loss = nn.MSELoss(reduction=self.reduction)


    def forward(self, output, gt):
        return self.rc_loss(gt, output)

     