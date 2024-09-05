try :
    from . import dg_sta
    from . import dg_sta_BN
    from . import generator
    from . import dg_sta_contrastive_trans_gen
    from . import dg_sta_contrastive
    from . import actor
    from . import dg_sta_var
    from . import stgcn
    from . import stgcn_backbone
    from . import dg_sta_backbone
    from .helpers import print_n_params
except :
    # import DFCIL.guided_MI.src.model_defs.dg_sta_contrastive_trans_gen as dg_sta_contrastive_trans_gen
    import dg_sta_contrastive
    import dg_sta
    import dg_sta_BN
    import generator
    import actor 
    import dg_sta_var
    import stgcn
    import DFCIL.guided_MI.src.model_defs.stgcn_backbone as stgcn_backbone
    import dg_sta_backbone
    from helpers import print_n_params

def get_model(cfg) :
    assert cfg.name in globals(), \
        f"Model {cfg.name} not found."

    if cfg.name == 'dg_sta' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            dropout=cfg.dropout, 
        )

    if cfg.name == 'dg_sta_BN' :
        print('Using dg_sta_BN model.')
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            dropout=cfg.dropout, 
        )

    if cfg.name == 'dg_sta_contrastive' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            d_head_contrastive=cfg.d_head_contrastive,
            dropout=cfg.dropout, 
        ), globals()[cfg.name].Classifier(
            n_classes=cfg.n_classes, 
            d_feat=cfg.d_feat, # feature dimension
        )
    if cfg.name == 'dg_sta_backbone' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            d_head_contrastive=cfg.d_head_contrastive,
            dropout=cfg.dropout, 
        ), globals()[cfg.name].Classifier(
            n_classes=cfg.n_classes, 
            d_feat=cfg.d_feat, # feature dimension
        )
    # if cfg.name == 'dg_sta_contrastive_trans_gen' :
    #     return globals()[cfg.name].Model(
    #         n_classes=cfg.n_classes, 
    #         in_channels=cfg.in_channels,
    #         n_heads=cfg.n_heads, # number of attention heads
    #         d_head=cfg.d_head, # dimension of attention heads
    #         d_feat=cfg.d_feat, # feature dimension
    #         seq_len=cfg.seq_len, # sequence length
    #         n_joints=cfg.n_joints, # number of joints
    #         d_head_contrastive=cfg.d_head_contrastive,
    #         dropout=cfg.dropout, 
    #     )
    if cfg.name == 'actor' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            d_head_contrastive=cfg.d_head_contrastive,
            dropout=cfg.dropout, 
        )
    if cfg.name == 'stgcn_backbone' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            graph_cfg=cfg.graph_cfg,
            gcn_adaptive=cfg.gcn_adaptive, 
            gcn_with_res=cfg.gcn_with_res, 
            tcn_type=cfg.tcn_type, 
            dropout=cfg.dropout, 
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
        ), globals()[cfg.name].Classifier(
             n_classes=cfg.n_classes, 
            d_feat=cfg.d_feat, # feature dimension
        )
    if cfg.name == 'stgcn' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            graph_cfg=cfg.graph_cfg,
            gcn_adaptive=cfg.gcn_adaptive, 
            gcn_with_res=cfg.gcn_with_res, 
            tcn_type=cfg.tcn_type, 
            dropout=cfg.dropout, 
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
        )
    
    if cfg.name == 'dg_sta_var' :
        return globals()[cfg.name].Model(
            n_classes=cfg.n_classes, 
            in_channels=cfg.in_channels,
            n_heads=cfg.n_heads, # number of attention heads
            d_head=cfg.d_head, # dimension of attention heads
            d_feat=cfg.d_feat, # feature dimension
            seq_len=cfg.seq_len, # sequence length
            n_joints=cfg.n_joints, # number of joints
            dropout=cfg.dropout, 
        )
    else :
        raise NotImplementedError


if __name__ == "__main__" :
    from helpers import print_n_params
    from easydict import EasyDict as edict

    import yaml
    import torch

    import sys 
    sys.path.append('..')
    from configs.datasets import hgr_shrec_2017

    root_dir = '/ogr_cmu/data/SHREC_2017'
    cfg_file = '../configs/params/initial.yaml'
    split_type = 'agnostic'
    cfg_data = hgr_shrec_2017.Config_Data(root_dir)

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model(edict({
            'n_classes': cfg_data.get_n_classes(split_type),
            **cfg_params.model,
    }))
    model = model.cuda(0)
    print(model)
    print_n_params(model)
    
    bs, seq_len, n_joints, in_channels = 4, 8, 22, 3
    x = torch.rand(bs, seq_len, n_joints, in_channels).cuda(0)
    out = model(x)
    print(x.shape, out.shape)