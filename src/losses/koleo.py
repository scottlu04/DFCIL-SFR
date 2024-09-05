from typing import  Optional, Sequence

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F 

try :
    from .helpers import one_hot
except :
    from helpers import one_hot

def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I

class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()
        self.pdist = nn.PairwiseDistance(2)
  


    def forward(self, feature):
        n = feature.shape[0]
    
        I = pairwise_NNs_inner(feature)
        distances = self.pdist(feature, feature[I])
        loss_uniform = - torch.log(n * distances).mean()
        return loss_uniform




