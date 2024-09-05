from typing import Optional

from easydict import EasyDict as edict
import math

try :
    from . import cross_entropy
    from . import contrastive
    from . import distillation
    from . import bce
    from . import abd_distillation
    from . import rkd
    from . import hkd
    from . import rdfcil_local_cross_entropy
    from . import abd_local_cross_entropy
    from . import mse
    from . import kl_loss
    from . import mmd
    from . import koleo
except :
    import cross_entropy    
    import DFCIL.guided_MI.src.losses.contrastive as contrastive
    import distillation
    import bce
    import abd_distillation
    import rkd
    import hkd
    import rdfcil_local_cross_entropy
    import abd_local_cross_entropy
    import mse
    import kl_loss
    import mmd
    import koleo

def _dict_to_list(d) :
    max_k = max(d.keys())
    assert len(d) == max_k + 1
    return [d[k] for k in range(max_k+1)]


def get_losses(cfg: dict, n_classes: int, class_freq: Optional[dict] = None) :
    criteria = edict()
    for lname in cfg :
        assert lname in globals(), f"{lname} not found."
        criteria[lname] = edict()
        criteria[lname].weight = cfg[lname].weight
        if lname == 'cross_entropy' :
            weights = None
            if cfg[lname].class_weight == 'class_freq' :
                raise NotImplementedError
                
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                n_classes,
                cfg[lname].ignore,
                weights,
            )
        
        elif lname == 'contrastive' :
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].n_views,
                cfg[lname].is_supervised,
                cfg[lname].temperature,
                cfg[lname].type,
            )  

        elif lname == 'distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'abd_distillation' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()  

        elif lname == 'bce' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].reduction
            )    
            
        elif lname == 'rkd' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(
                cfg[lname].in_dim1,
                cfg[lname].in_dim2,
                cfg[lname].proj_dim,
            )
        elif lname == 'hkd' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    

        elif lname == 'rdfcil_local_cross_entropy' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()   

        elif lname == 'abd_local_cross_entropy' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    
        elif lname == 'mse' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    
        elif lname == 'kl_loss' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')()    
        elif lname == 'mmd' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')(n_classes)   
        elif lname == 'koleo' :   
            criteria[lname].func = getattr(globals()[lname], 'Loss')() 
        else : 
            raise NotImplementedError

    return criteria


if __name__ == "__main__" :
    def gen_data(bs, c, h, w) :
        import torch
        xs = torch.rand(bs, c, h, w)
        ys = torch.randint(0, c, (bs, ))      
        ms = torch.rand(bs, h, w) > 0.5
        xs.requires_grad_(True)
        return xs, ys, ms

    import yaml
    from pprint import pprint

    import sys
    sys.path.append('..')
    from configs.datasets import hgr_shrec_2017

    root_dir = '/data/datasets/agr/shrec2017'
    cfg_file = '../configs/params/initial.yaml'
    split_type = 'single'
    cfg_data = hgr_shrec_2017.Config_Data(root_dir)

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    n_classes = cfg_data.get_n_classes(split_type)

    criteria = get_losses(
                    cfg_params.loss, 
                    n_classes, 
    )

    pprint(criteria)
    print()

    bs, c, h, w = 4, n_classes, 1, 1
    logits, label, mask = gen_data(bs, c, h, w)    

    for k in criteria :
        print(k, criteria[k].func(logits, label, mask=None))

