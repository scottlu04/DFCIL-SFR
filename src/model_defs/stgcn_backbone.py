# Source: https://github.com/kennymckormick/pyskl/tree/main

import copy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.runner import load_checkpoint

# from ...utils import Graph, cache_checkpoint

try :
    from .helpers import *
    from .gcn_utils import mstcn, unit_gcn, unit_tcn, Graph
except:
    from helpers import *
    from gcn_utils import mstcn, unit_gcn, unit_tcn,Graph

EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class Model(nn.Module):

    def __init__(self,
                 
                 graph_cfg,
                 n_classes=60,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 dropout=0.5,
                 d_feat=256,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained


        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc_cls = nn.Linear(256, 60)
        # self.mlp1  =nn.Sequential(nn.Linear(d_feat, d_feat), nn.ReLU(),
        #                     nn.Linear(d_feat, d_feat))
        self.mlp  =nn.Sequential(nn.Linear(d_feat, d_feat), nn.ReLU(),
                            nn.Linear(d_feat, d_feat))
        
    # def init_weights(self):
    #     if isinstance(self.pretrained, str):
    #         self.pretrained = cache_checkpoint(self.pretrained)
    #         load_checkpoint(self, self.pretrained, strict=False)

    def forward_feature(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])

        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = self.pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        # if self.dropout is not None:
        #     x = self.dropout(x)

        feature_cl = self.mlp(x)
        feature_cl = F.normalize(feature_cl,dim=1)

        # cls_score = self.fc_cls(x)

        return (x, feature_cl)

class Classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, 
        n_classes: int,
        d_feat: int,
        ):
        super(Classifier,self).__init__()
        # self.fc1 = nn.Linear(d_feat,d_feat*2)
        self.fc = nn.Linear(d_feat, n_classes)
        self.mlp  =nn.Sequential(nn.Linear(d_feat, d_feat), nn.ReLU(),
                            nn.Linear(d_feat, n_classes))

    def forward(self, features):
        #x = self.mlp(features)
        x = self.fc(features)
        return x
    
    def update_fc_avg(self,proto_dict):

        for idx in proto_dict:
            print(idx)
            print(self.fc.weight.data[idx].shape)
            device = self.fc.weight.data[idx].device
            proto  = torch.from_numpy(proto_dict[idx]).cuda(device)
            print(proto.shape)
            # self.fc.weight.data[idx ]=torch.squeeze(proto)
            self.fc.weight.data[idx ]=F.normalize(torch.squeeze(proto), p=2, dim=-1)
        print('end')

    def soft_calibration(self, cfg, current_t_index):
        softmax_t = cfg.increm.learner.softmax_t
        shift_weight = cfg.increm.learner.shift_weight

        base_protos = self.fc.weight.data[:cfg.increm.first_split_size].detach().cpu().data
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        
        print(cfg.increm.first_split_size + (current_t_index-1) * cfg.increm.other_split_size,  
                                         cfg.increm.first_split_size + current_t_index * cfg.increm.other_split_size)
        cur_protos = self.fc.weight.data[cfg.increm.first_split_size + (current_t_index-1) * cfg.increm.other_split_size : 
                                         cfg.increm.first_split_size + current_t_index * cfg.increm.other_split_size].detach().cpu().data
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        
        weights = torch.mm(cur_protos, base_protos.T) * softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)

        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        
        updated_protos = (1-shift_weight) * cur_protos + shift_weight * delta_protos

        self.fc.weight.data[cfg.increm.first_split_size + (current_t_index-1) * cfg.increm.other_split_size : 
                            cfg.increm.first_split_size + current_t_index * cfg.increm.other_split_size] = updated_protos
        # new_fc=torch.stack(new_fc,dim=0)
        # return new_fc

if __name__ == "__main__" :
    # graph_cfg=dict(layout='nturgb+d', mode='spatial')
    # model = dict(
    # type='RecognizerGCN',
    # backbone=dict(
    #     type='STGCN',
    #     gcn_adaptive='init',
    #     gcn_with_res=True,
    #     tcn_type='mstcn',
    #     graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    # cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))

    model = Model(graph_cfg=dict(layout='nturgb+d', mode='spatial'),
                gcn_adaptive='init', gcn_with_res=True,
                     tcn_type='mstcn')
    model = model.cuda(0);


    # inputs = torch.randn(batch_size, num_person,
    #                         num_frames, num_joints, 3)
    bs, num_person, seq_len, n_joints, in_channels = 4, 2, 20, 25, 3;
    x = torch.rand(bs,num_person, seq_len, n_joints, in_channels).cuda(0);
    out = model(x);
    print(x.shape, out.shape);
    
    # model = Model(model['backbone'])
    # from helpers import print_n_params
    # from easydict import EasyDict as edict

    # import yaml

    # import sys; sys.path.append('..');
    # from configs.datasets import hgr_shrec_2017

    # root_dir = '/data/datasets/agr/shrec2017';
    # cfg_file = '../configs/params/initial.yaml';
    # split_type = 'specific';
    # cfg_data = hgr_shrec_2017.Config_Data(root_dir);

    # with open(cfg_file, 'rb') as f :
    #     cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

    # print(edict({
    #         'n_classes': cfg_data.get_n_classes(split_type),
    #         **cfg_params.model,
    # }))

    # model = Model(edict({
    #         'n_classes': cfg_data.get_n_classes(split_type),
    #         **cfg_params.model,
    # }));
    # model = model.cuda(0);
    # print(model);
    # print_n_params(model);
    
    # bs, seq_len, n_joints, in_channels = 4, 8, 22, 3;
    # x = torch.rand(bs, seq_len, n_joints, in_channels).cuda(0);
    # out = model(x);
    # print(x.shape, out.shape);