import torch 
import torch.nn as nn
import torch.nn.functional as F 

# def one_hot(x: torch.Tensor, n_classes: int) -> torch.Tensor :
#     assert torch.is_tensor(x)
#     shape = x.shape
#     oh = torch.zeros(
#             (shape[0], n_classes) + shape[1:], 
#             device=x.device, 
#             dtype=x.dtype, 
#     )

#     return oh.scatter_(1, x.unsqueeze(1).long(), 1.)

def one_hot(target,num_class):
    one_hot = torch.zeros(target.shape[0],num_class).cuda()
    one_hot = one_hot.scatter(1,target.long().view(-1,1),1.)
    return one_hot