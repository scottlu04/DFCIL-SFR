"""Modified from source: https://github.com/HobbitLong/SupContrast/blob/master/losses.py """

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from typing import Optional, Sequence

def _assert_isnan(x: Tensor) : 
    assert not torch.any(torch.isnan(x))


class Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, 
        n_views: int,
        is_supervised: bool,
        temperature: float = 0.07, 
        type: str = 'triplet'
    ) -> None :

        super().__init__()

        assert is_supervised, \
            "Implementation not checked for unsupervised computations."

        assert type in ['triplet', 'infonce']
        self.n_views = n_views
        self.temperature = temperature
        self.is_supervised = is_supervised
        self.type = type 
        
        print('using ' + type +" as loss function")
        # for triplet loss only
        self.margin = 0.5 
        self.alpha = 0.5 
        
        # self.incremental_alpha = 0.9


    def forward(self, feature, target,new_idx,old_idx):
        if self.type == "triplet":
            return self.forward_triplet(feature, target,new_idx,old_idx)
        elif self.type == "infonce":
            return self.forward_infonce(feature, target)


    def forward_triplet(self, feature, target,new_idx,old_idx):
        #TODO 
        if old_idx.shape[0] != 0:
            self.alpha = 0.9 
        inputs_col, targets_col = feature, target
        inputs_row, target_row = feature, target
    
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()
        # print(target)
        neg_count = list()
        for i in range(n):
            # print(i+1)
            # print(sim_mat[i])
            if i < new_idx.shape[0]:
                    pos_pair_ = torch.masked_select(sim_mat[i][new_idx], targets_col[i] == target_row[new_idx])
                    # print('pos')
                    # print(targets_col[i] == target_row)
                    # print(pos_pair_)
                    pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
                    neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)
                    neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
                    # print('neg')
                    # print(targets_col[i] != target_row)
                    # print(neg_pair_)
                    
            else:
                    neg_pair_ = torch.masked_select(sim_mat[i][new_idx], targets_col[i] != target_row[new_idx])
                    neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
                    # print('neg')
                    # print(targets_col[i] != target_row[new_idx])
                    # print(neg_pair_)
            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
            else:
                    neg_loss = 0
            # loss.append(pos_loss + neg_loss)

            loss.append(self.alpha * pos_loss +(1-self.alpha) * neg_loss)

        loss = sum(loss) / n  # / all_targets.shape[1]
        # print(loss)
        return loss

    def forward_infonce(self, 
        features: Tensor, 
        labels: Optional[Tensor] = None,  
        mask: Optional[Tensor] = None,
    ) -> Tensor :

        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        dtype = features.dtype
      
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).to(dtype).to(device)
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # bs * nv x d
        contrast_feature = features
        if self.is_supervised :
            anchor_feature = contrast_feature
            anchor_count = self.n_views
        else :
            anchor_feature = features[:, 0]
            anchor_count = 1


        # compute logits
        anchor_dot_contrast = (anchor_feature @ contrast_feature.T) / self.temperature # (bs * nv) x (bs * nv)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max #logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, self.n_views)
        # mask-out self-contrast cases
        logits_mask = (1 - torch.eye(mask.size(0))).to(device)  
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.clamp(exp_logits, min=1e-3)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()
        _assert_isnan(loss)
        return loss


if __name__ == "__main__" :
    def gen_data(bs, nv, d, c) :
        xs = torch.rand(bs, nv, d)
        ys = torch.randint(0, c, (bs, ))      
        xs.requires_grad_(True)
        return xs, ys

    bs, nv, d, c = 16, 2, 16, 10
    logits, label = gen_data(bs, nv, d, c)
    print(label)
    loss = Loss(nv, True)(logits, label)
    loss.backward()
    print(loss)    