from easydict import EasyDict as edict
import inspect
from .transforms import *

def get_transforms_from_cfg(cfg) :
    if cfg is None :
        return None

    print(cfg)
    tlist = []
    for ttype in cfg :
        tlist.append(globals()[ttype](cfg[ttype]))

    # for ttype in cfg:
    #     func = globals()[ttype]  # Get the function based on ttype
    #     # Check if the function expects any arguments
    #     params = inspect.signature(func).parameters
    #     if params:  # If the function expects arguments, pass cfg[ttype]
    #         tlist.append(func(**cfg[ttype]))
    #         # print(ttype)
    #         # print(cfg[ttype])
    #     else:  # If the function expects no arguments, call it without arguments
    #         tlist.append(func())

    
    if len(tlist) == 1 :
        return tlist[0]

    return Compose(tlist)

def get_transforms_from_cfg_ntu(cfg) :
    if cfg is None :
        return None

    # print(cfg)
    tlist = []
    # for ttype in cfg :
    #     tlist.append(globals()[ttype](cfg[ttype]))

    for ttype in cfg:
        func = globals()[ttype]  # Get the function based on ttype
        # Check if the function expects any arguments
        params = inspect.signature(func).parameters
        if params:  # If the function expects arguments, pass cfg[ttype]
            tlist.append(func(**cfg[ttype]))
            # print(ttype)
            # print(cfg[ttype])
        else:  # If the function expects no arguments, call it without arguments
            tlist.append(func())

    
    if len(tlist) == 1 :
        return tlist[0]

    return Compose(tlist)

def random_scale(cfg) :
    assert 'lim' in cfg
    return RandomScale(cfg.lim)


def random_noise(cfg) :
    assert 'lim' in cfg
    rm_global_scale = cfg.rm_global_scale if 'rm_global_scale' in cfg else False
    return RandomNoise(cfg.lim, rm_global_scale)


def random_translation(cfg) :
    assert 'x' in cfg
    assert 'y' in cfg
    assert 'z' in cfg
    return RandomTranslation(cfg.x, cfg.y, cfg.z)


def random_rotation(cfg) :
    assert 'x' in cfg
    assert 'y' in cfg
    assert 'z' in cfg
    return RandomRotation(cfg.x, cfg.y, cfg.z)


def random_time_interpolation(cfg) :
    assert 'prob' in cfg
    return RandomTimeInterpolation(cfg.prob)                


def stratified_sample(cfg) :
    assert 'n_samples' in cfg
    return StratifiedSample(cfg.n_samples)


def center_by_index(cfg) :
    assert 'ind' in cfg
    n_hands = cfg.n_hands if 'n_hands' in cfg else 1
    return CenterByIndex(cfg.ind, n_hands)