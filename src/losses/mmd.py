# -*- coding: utf-8 -*-

"""
Pytorch port of Relation Knowledge Distillation Losses.

credits:
    https://github.com/lenscloth/RKD/blob/master/metric/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]

def KL(mu,std):
    l2_z_mean= LA.norm(mu.mean(dim=0), ord=2)
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return KL
def get_vector_property(x):
    N, C = x.size()
    x1 = x.unsqueeze(0).expand(N, N, C)
    x2 = x.unsqueeze(1).expand(N, N, C)
    x1 = x1.reshape(N*N, C)
    x2 = x2.reshape(N*N, C)
    cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-6).view(N, N)
    cos_sim = torch.triu(cos_sim, diagonal=1).sum() * 2 / (N*(N-1))
    pdist = (LA.norm(x1-x2, ord=2, dim=1)).view(N, N)
    pdist = torch.triu(pdist, diagonal=1).sum() * 2 / (N*(N-1))
    return cos_sim, pdist

# class RKDAngleLoss(nn.Module):
class Loss(nn.Module):
    def __init__(
        self,n_class
    ):
        super().__init__()
        self.n_class = n_class
        self.cls_loss = LabelSmoothingCrossEntropy()

    # def forward(self,
    #     logits: torch.Tensor,
    #     target: torch.Tensor,
    #     z: torch.Tensor,
    #     z_prior: torch.Tensor,
    #     mu,
    #     std,
    #     ):
    #     # N x C
    #     # N x N x C
    #     # print(self.n_class)
    #     mmd_loss, l2_z_mean, z_mean = get_mmd_loss(z, z_prior, target,self.n_class )
    #     # cos_z, dis_z = get_vector_property(z_mean)
    #     # cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
    #     # cos_z_value.append(cos_z.data.item())
    #     # dis_z_value.append(dis_z.data.item())
    #     # cos_z_prior_value.append(cos_z_prior.data.item())
    #     # dis_z_prior_value.append(dis_z_prior.data.item())
    #     lambda_1=1e-4
    #     lambda_2=1e-1
    #     # z_prior_gain=3
    #     cls_loss = self.cls_loss(logits, target)
    #     loss = lambda_2* mmd_loss + lambda_1* l2_z_mean + cls_loss
    #     return loss
    

    def forward(self,
        logits: torch.Tensor,
        target: torch.Tensor,
        z: torch.Tensor,
        z_prior: torch.Tensor,
        mu,
        std,
        ):
        # N x C
        # N x N x C
        kl = KL(mu,std)
        # cos_z, dis_z = get_vector_property(z_mean)
        # cos_z_prior, dis_z_prior = get_vector_property(self.model.z_prior)
        # cos_z_value.append(cos_z.data.item())
        # dis_z_value.append(dis_z.data.item())
        # cos_z_prior_value.append(cos_z_prior.data.item())
        # dis_z_prior_value.append(dis_z_prior.data.item())
        lambda_1=1e-4
        lambda_2=1e-5
        # z_prior_gain=3
        cls_loss = self.cls_loss(logits, target)
        loss = lambda_2* kl + cls_loss
        return loss


