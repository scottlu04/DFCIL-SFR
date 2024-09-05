# Source: https://github.com/yuxiaochen1103/DG-STA/blob/60552d58856bd9728f1b249731ff23148fb30701/model/network.py 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import numpy as np

from functools import partial

try :
    from .helpers import *
except:
    from helpers import *


DOM_TYPE_ALL = ('spatial', 'temporal');


def _assert_dom_type(dom_type: str) :
    assert dom_type in DOM_TYPE_ALL, \
        f"Domain type must be one of {DOM_TYPE_ALL}, but got {dom_type}";      

class PositionalEncoding_decode(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding_decode, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class PositionalEncoding(nn.Module) :
    def __init__(self, 
        n_joints: int,
        seq_len: int, 
        d_model: int,               
        dom_type: str,
    ) -> None:

        super().__init__();

        _assert_dom_type(dom_type);

        self.n_joints = n_joints;
        self.seq_len = seq_len;
        self.dom_type = dom_type;

        dtype = torch.get_default_dtype();

        #temporal positial embedding
        if dom_type == 'temporal':
            pos_list = list(range(self.n_joints * self.seq_len));

        # spatial positial embedding
        elif dom_type == 'spatial' :
            pos_list = [];
            for t in range(self.seq_len) :
                for j_id in range(self.n_joints) :
                    pos_list.append(j_id);

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).to(dtype);

        # Compute the positional encodings once in log space
        pe = torch.zeros(self.seq_len * self.n_joints, d_model);
        div_term = torch.exp(torch.arange(0, d_model, 2).to(dtype) *
                             ( -math.log(10000.0) / d_model ));
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term);
        pe.unsqueeze_(0);
        self.register_buffer('pe', pe);


    def forward(self, x: Tensor ) -> Tensor :
        """
        Args:
        x: Tensor, shape [?]
        """
        x = x + self.pe[:, :x.size(1)];
        return x;



class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, 
        d_model: int, 
        eps: float = 1e-6,
    ) -> None :

        super().__init__();

        self.a_2 = nn.Parameter(torch.ones(d_model));
        self.b_2 = nn.Parameter(torch.zeros(d_model));
        self.eps = eps;


    def forward(self, x: Tensor) -> Tensor :
        """
        Args:
        x: Tensor, shape [batch_size, seq_len, d_model]
        """        
        mean = x.mean(-1, keepdim=True);
        std = x.std(-1, keepdim=True);
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2;
        return x;


class QkvProjector(nn.Module) :
    """Project Q, K, V for onto desired attention blocks."""
    def __init__(self, 
        d_in: int,
        n_heads: int,
        d_head: int,
    ) -> None :

        super().__init__();

        self.n_heads = n_heads;
        self.d_head = d_head;

        self.projector = nn.Linear(d_in, self.n_heads * self.d_head);

    def forward(self, x: Tensor) -> Tensor :
        bs = x.size(0);
        x = self.projector(x); # bs x n x n_h * d_h
        x = x.view(bs, -1, self.n_heads, self.d_head); # bs x n x n_h x d_h
        x = x.transpose(1, 2); # bs x n_h x n x d_h                

        return x;



class MultiHeadedAttention(nn.Module) :
    def __init__(self, 
        n_heads: int, 
        d_head: int, 
        d_in: int, 
        dom_type: str,
        seq_len: int,
        n_joints: int,
        dropout: float = 0.,
        eps: float = 1e-6,
    ) -> None :

        super().__init__()

        _assert_dom_type(dom_type);

        self.d_head = d_head;
        self.n_heads = n_heads;
        self.dom_type = dom_type;
        self.seq_len = seq_len;
        self.n_joints = n_joints;

        self.d_model = self.n_heads * self.d_head;        
        self.eps = eps;

        st_mask = self.get_spatiotemporal_mask();
        self.register_buffer('scaled_st_mask', st_mask / math.sqrt(self.d_head));
        self.register_buffer('th_logit', 
                (1 - st_mask) * torch.finfo(torch.get_default_dtype()).min );       

        if dropout < 1e-3 :
            self.k_map = QkvProjector(d_in, self.n_heads, self.d_head);
            self.q_map = QkvProjector(d_in, self.n_heads, self.d_head);
            self.v_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.ReLU(),
            );

        else :
            self.k_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.Dropout(dropout),
            );

            self.q_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.Dropout(dropout),
            );

            self.v_map = nn.Sequential(
                            QkvProjector(d_in, self.n_heads, self.d_head),
                            nn.ReLU(),
                            nn.Dropout(dropout),
            );            


    def get_spatiotemporal_mask(self) :
        # Sec 3.4
        n_pts = self.seq_len * self.n_joints;
        s_mask = torch.zeros(n_pts, n_pts);

        for i in range(self.seq_len) :
            i_begin = i * self.n_joints;
            i_end = i_begin + self.n_joints;
            # t_mask[i_begin: i_end, i_begin: i_end].fill_(0); #Sec 3.4
            s_mask[i_begin:i_end, i_begin:i_end].fill_(1); #Sec 3.4

        if self.dom_type == 'spatial' :
            return s_mask;

        t_mask = 1 - s_mask + torch.eye(n_pts);
        return t_mask;
    

    def compute_qkv_attention(self, 
        q: Tensor, # bs x n_h x n x d_h
        k: Tensor, # bs x n_h x n x d_h 
        v: Tensor, # bs x n_h x n x d_h
    ) -> Tensor :

        scores = q @ k.transpose(-2, -1); # bs x n_h x n x n
        scores = (scores * self.scaled_st_mask) + self.th_logit;

        tmp_max, _ = scores.max(dim=-1, keepdim=True);
        scores = torch.exp(scores - tmp_max);
        scores = scores / (scores.sum(dim=-1, keepdim=True) + self.eps);

        out = scores @ v; # bs x n_h x n x d_h
        return out;


    def forward(self, x):
        """
        Args :
        x : Tensor, shape [bs x n x n_h * d_h]
        Returns :
        x : Tensor, shape [bs x n x n_h * d_h]
        """
        bs = x.size(0); # bs x n x n_h * d_h

        q = self.q_map(x); # bs x n_h x n x d_h
        k = self.k_map(x); # bs x n_h x n x d_h
        v = self.v_map(x); # bs x n_h x n x d_h

        x = self.compute_qkv_attention(q, k, v); # bs x n_h x n x d_h
        x = x.transpose(1, 2).contiguous(); # bs x n x n_h x d_h
        x = x.view(bs, -1, self.d_model); # bs x n x n_h * d_h

        return x;


class SpatioTemporalAttention(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, 
        d_in: int, # input
        d_out: int, # output 
        n_heads: int, # number of attention heads
        d_head: int, # dimension of attention heads
        seq_len: int, # sequence length
        n_joints: int, # number of joints
        dom_type: str, # 'spatial' or 'temporal' domain
        dropout: float = 0., 
    ) -> None :

        super().__init__();

        self.pe = PositionalEncoding(n_joints, seq_len, d_in, dom_type);

        self.att_layer = MultiHeadedAttention(
                        n_heads, 
                        d_head, 
                        d_in, 
                        dom_type,
                        seq_len,
                        n_joints,
                        dropout, 
        ); 

        layers = [
            nn.Linear(n_heads * d_head, d_out),
            nn.ReLU(),
            LayerNorm(d_out),
        ];

        if dropout < 1e-3 : 
            layers.append(nn.Dropout(dropout));

        self.linear = nn.Sequential(*layers);

        self.init_parameters();


    def forward(self, 
        x: Tensor,
    ) -> Tensor :

        x = self.pe(x); #add PE
        x = self.att_layer(x); #pass attention model
        x = self.linear(x); 
        return x;


    def init_parameters( self ) :
        model_list = [ self.att_layer, self.linear];
        for model in model_list :
            for p in model.parameters() :
                if p.dim() > 1 :
                    nn.init.xavier_uniform_(p);


class ContrastiveHead(nn.Module) :
    def __init__(self,
        d_in: int,
        d_out: int,
    ) -> None :

        super().__init__();

        self.layers = nn.Sequential(
                nn.Linear(d_in, d_in),
                nn.ReLU(),
                nn.Linear(d_in, d_out),
        );        

    def forward(self, x: Tensor) -> Tensor :
        return F.normalize(self.layers(x), dim=1);


class Encoder(nn.Module):
    def __init__(self, 
        n_classes: int, 
        in_channels: int, 
        n_heads: int, # number of attention heads
        d_head: int, # dimension of attention heads
        d_feat: int, # feature dimension
        seq_len: int, # sequence length
        n_joints: int, # number of joints
        d_head_contrastive: int,
        dropout: float = 0, 
    ) -> None :

        super().__init__();

        self.seq_len = seq_len;
        self.n_joints = n_joints;
        self.n_pts = seq_len * n_joints;
        self.in_channels = in_channels;

        st_layer = partial(SpatioTemporalAttention,
                        d_in=d_feat,
                        d_out=d_feat,
                        n_heads=n_heads,
                        d_head=d_head,
                        seq_len=seq_len,
                        n_joints=n_joints,
                        dropout=dropout,
        );

        self.initial = nn.Sequential(
                nn.Linear(in_channels, d_feat),
                nn.ReLU(),
                LayerNorm(d_feat),
                nn.Dropout(dropout),
        );
        self.spatial_att = st_layer(dom_type='spatial');
        self.temporal_att = st_layer(dom_type='temporal');
        self.final = nn.Linear(d_feat, n_classes);

        # self.contrastive_head = ContrastiveHead(d_feat, d_head_contrastive);
        self.contrastive_head = nn.Identity();


    def get_input_shape(self) :
        return [self.seq_len, self.n_joints, self.in_channels];


    def forward_feature(self, 
        x: Tensor, # bs x seq_len x n_joints x in_channels
    ) -> Tensor :

        assert not torch.any(torch.isnan(x));
        x = x.view(-1, self.n_pts, self.in_channels);
        x = self.initial(x);
        x = self.spatial_att(x);
        x = self.temporal_att(x); # bs x seq_len * n_joints x d_out
        x = torch.mean(x, dim=1); # bs x d_out
        return x; 


    def forward(self, 
        x: Tensor, # bs x seq_len x n_joints x in_channels
    ) -> Tensor :

        x = self.forward_feature(x); # bs x d_out
        x = self.final(x); # bs x n_classes
        return x;


    def forward_contrastive(self, 
        x: Tensor, # bs x seq_len x n_joints x in_channels
        add_cls: bool = False,
    ) -> Tensor :

        x = self.forward_feature(x); # bs x d_out
        x_contrastive = self.contrastive_head(x);
        if not add_cls :
            return x_contrastive;

        x = self.final(x);
        return x_contrastive, x;           


    def classify(self, 
        x: Tensor, # bs x d_out
    ) -> Tensor :

        return self.final(x);


    def pretrain(self) :
        self.initial.train(); self.initial.requires_grad_(True);
        self.spatial_att.train(); self.spatial_att.requires_grad_(True);
        self.temporal_att.train();  self.temporal_att.requires_grad_(True);
        self.contrastive_head.train(); self.contrastive_head.requires_grad_(True);
        self.final.eval();  self.final.requires_grad_(False);

    def train_w_grad(self) :
        self.initial.train(); self.initial.requires_grad_(True);
        self.spatial_att.train(); self.spatial_att.requires_grad_(True);
        self.temporal_att.train();  self.temporal_att.requires_grad_(True);
        self.contrastive_head.train(); self.contrastive_head.requires_grad_(True);
        self.final.train();  self.final.requires_grad_(True);


    def finetune(self) :
        self.initial.eval(); self.initial.requires_grad_(False);
        self.spatial_att.eval(); self.spatial_att.requires_grad_(False);
        self.temporal_att.eval();  self.temporal_att.requires_grad_(False);
        self.contrastive_head.eval(); self.contrastive_head.requires_grad_(False);
        self.final.train();  self.final.requires_grad_(True);

    def reset_classifier(self) :
        n_in, n_out = self.final.in_features, self.final.out_features;
        device = self.final.weight.device;
        self.final = nn.Linear(n_in, n_out).to(device);
        self.final.train(); self.final.requires_grad_(True);


    def eval_feature(self) :
        self.initial.eval(); self.initial.requires_grad_(True);
        self.spatial_att.eval(); self.spatial_att.requires_grad_(True);
        self.temporal_att.eval();  self.temporal_att.requires_grad_(True);
        self.contrastive_head.eval(); self.contrastive_head.requires_grad_(True);
        self.final.eval();  self.final.requires_grad_(False);        

class Model(nn.Module):
    def __init__(self, 
        n_classes: int, 
        in_channels: int, 
        n_heads: int, # number of attention heads
        d_head: int, # dimension of attention heads
        d_feat: int, # feature dimension
        seq_len: int, # sequence length
        n_joints: int, # number of joints
        d_head_contrastive: int,
        dropout: float = 0, 
    ) -> None :

        super().__init__();

        self.seq_len = seq_len;
        self.n_joints = n_joints;
        self.n_pts = seq_len * n_joints;
        self.in_channels = in_channels;
        self.d_model = n_heads * d_head
        self.encoder = Encoder( 
                    n_classes = n_classes, 
                    in_channels = in_channels,
                    n_heads = n_heads,
                    d_head = d_head,
                    d_feat = d_feat,
                    seq_len = seq_len,
                    n_joints = n_joints,
                    d_head_contrastive = d_head_contrastive,
                    dropout = dropout,
        );

        self.mu_layer = nn.Linear(d_feat, self.d_model)
        self.sigma_layer = nn.Linear(d_feat, self.d_model)
        self.actionBiases = nn.Parameter(torch.randn(n_classes, self.d_model))

        self.sequence_pos_encoder = PositionalEncoding_decode(self.d_model, dropout)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                          nhead=n_heads,
                                                          dim_feedforward=d_feat,
                                                          dropout=dropout)
                                                         # activation=activation)
        self.Decoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=1)

        self.finallayer = nn.Linear(self.d_model, self.n_joints* self.in_channels)

    def reparameterize(self, mu, logvar, seed=None):
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z
    def decode(self, 
        z: Tensor, # 
        y
    ) -> Tensor :
        bs = z.size(0)
        z = z + self.actionBiases[y]
        z =z[None]
        #print(z.size())
        timequeries = torch.zeros(self.seq_len, bs, self.d_model, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        output = self.Decoder(tgt=timequeries, memory=z)

        output = self.finallayer(output).reshape(self.seq_len, bs, self.n_joints, self.in_channels)
        
        # zero for padded area
        output = output.permute(1, 0, 2, 3)
        
        #print(output.size())
        #x = self.encoder.final(x); # bs x n_classes
        return output
    
    def forward_contrastive(self, 
        x: Tensor, # bs x seq_len x n_joints x in_channels
        y
    ) -> Tensor :

        z = self.encoder.forward_feature(x); # bs x d_out
        #print(x.size())
        bs = x.size(0)
        #print(bs)
        mu =  self.mu_layer(z)
        var =  self.sigma_layer(z)
        z = self.reparameterize(mu,var)
        output = self.decode(z,y)
        return output, mu, var
    
    
    def generate(self):
        classes = torch.tensor([0,1,2,3,4,5,6,7])
        nats = len(classes)
        nspa = 5
        z = torch.randn(nspa*nats, self.d_model).cuda()#[None]
        y = classes.repeat(nspa)
        out = self.decode(z,y)
        return out,y
            
    # def generate(self, classes, durations, nspa=1,
    #              noise_same_action="random", noise_diff_action="random",
    #              fact=1):
    #     if nspa is None:
    #         nspa = 1
    #     nats = len(classes)
            
    #     y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))

    #     if len(durations.shape) == 1:
    #         lengths = durations.to(self.device).repeat(nspa)
    #     else:
    #         lengths = durations.to(self.device).reshape(y.shape)
        
    #     mask = self.lengths_to_mask(lengths)
        
    #     if noise_same_action == "random":
    #         if noise_diff_action == "random":
    #             z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
    #         elif noise_diff_action == "same":
    #             z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
    #             z = z_same_action.repeat_interleave(nats, axis=0)
    #         else:
    #             raise NotImplementedError("Noise diff action must be random or same.")
    #     elif noise_same_action == "interpolate":
    #         if noise_diff_action == "random":
    #             z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
    #         elif noise_diff_action == "same":
    #             z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
    #         else:
    #             raise NotImplementedError("Noise diff action must be random or same.")
    #         interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
    #         z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*nats, -1)
    #     elif noise_same_action == "same":
    #         if noise_diff_action == "random":
    #             z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
    #         elif noise_diff_action == "same":
    #             z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
    #         else:
    #             raise NotImplementedError("Noise diff action must be random or same.")
    #         z = z_diff_action.repeat((nspa, 1))
    #     else:
    #         raise NotImplementedError("Noise same action must be random, same or interpolate.")

    #     batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
    #     batch = self.decoder(batch)
        
    #     if self.outputxyz:
    #         batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
    #     elif self.pose_rep == "xyz":
    #         batch["output_xyz"] = batch["output"]
        
    #     return batch
# if __name__ == "__main__" :
#     from helpers import print_n_params
#     from easydict import EasyDict as edict

#     import yaml

#     import sys; sys.path.append('..');
#     from configs.datasets import hgr_shrec_2017

#     root_dir = '/data/datasets/agr/shrec2017';
#     cfg_file = '../configs/params/initial.yaml';
#     split_type = 'specific';
#     cfg_data = hgr_shrec_2017.Config_Data(root_dir);

#     with open(cfg_file, 'rb') as f :
#         cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

#     print(edict({
#             'n_classes': cfg_data.get_n_classes(split_type),
#             **cfg_params.model,
#     }))

#     model = Model(edict({
#             'n_classes': cfg_data.get_n_classes(split_type),
#             **cfg_params.model,
#     }));
#     model = model.cuda(0);
#     print(model);
#     print_n_params(model);
    
#     bs, seq_len, n_joints, in_channels = 4, 8, 22, 3;
#     x = torch.rand(bs, seq_len, n_joints, in_channels).cuda(0);
#     out = model(x);
#     print(x.shape, out.shape);