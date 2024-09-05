import sys
import os, os.path as osp

from easydict import EasyDict as edict
from typing import Any, Callable, Sequence, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

try :
    from .base import Dataset as BaseDataset
    from .helpers import *
    from . import transforms
except :
    from base import Dataset as BaseDataset
    from helpers import *
    import transforms    

sys.path.append('..')

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *



class Dataset(BaseDataset) :
    def __init__(self,
        mode: str,
        split_type: str,
        cfg: Config_Data,
        cfg_xforms: dict, 
        n_add_classes: int = -1,
        n_known_classes: int = 0,
        rm_global_scale: bool = False,
        drop_seed: int = -1,
        few_shot_seed: int = -1,
        few_shot_size: int = -1,
    ) -> None :

        super().__init__(
            mode, 
            cfg.get_split_filepath(split_type, mode), 
            cfg, 
            n_add_classes, 
            n_known_classes, 
            cfg.get_n_classes(split_type),
            rm_global_scale,
            drop_seed,
            few_shot_seed,
            few_shot_size
        )

        cfg_xforms = edict(cfg_xforms)
        self.xforms = transforms.get_transforms_from_cfg(cfg_xforms)
        self.to_tensor = transforms.ToTensor()

    def __getitem_trainval(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        pts = self.xforms(pts)
        pts = self.to_tensor(pts)

        data = edict({
            'pts': pts,
            'label': int(label),
        })

        return data

    def __getitem_test(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        pts = self.xforms(pts)
        pts = self.to_tensor(pts)

        rel_path = pts_path[len(self.cfg.root_dir)+1:]
        rel_path = osp.dirname(rel_path).replace('/', '__')
        data = edict({
            'path': rel_path,
            'pts': pts,
            'label': label,
        })

        return data        


    def __getitem__(self, idx) :
        if (self.mode == 'train') \
            or (self.mode == 'val') :
            return self.__getitem_trainval(idx)
        elif self.mode.startswith('test') :
            return self.__getitem_test(idx)
        else :
            raise NotImplementedError


if __name__ == "__main__" :

    test_loader = False
    test_time = False


    import yaml, random
    from pprint import pprint
    from tqdm import tqdm
    from configs.datasets import hgr_shrec_2017
    from utils.colors import *

    # root_dir = '/data/datasets/agr/shrec2017'
    root_dir = "/media/exx/HDD/zhenyulu/HandGestureDataset_SHREC2017";
    cfg_file = '/home/luzhenyu/DFCIL/guided_MI/src/configs/params/hgr_shrec_2017/Base.yaml';
    split_type = 'agnostic';
    cfg_data = hgr_shrec_2017.Config_Data(root_dir)

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader))

    mode = 'train'
    # mode = 'val'
    dataset = Dataset(
                mode, 
                split_type,
                cfg_data,
                cfg_params.transforms[mode],
                n_add_classes= 10,
                n_known_classes = 0,
                drop_seed=1,
              
    )
    print(len(dataset))
    class_map = {}
    mean_sample_dict = {}
    for i in range(len(dataset)):
        #print(gt.shape)
        label = dataset[i]['label']
        if label not in class_map:
            pts = np.array(dataset[i]['pts'])
            class_map[label] = [pts]
        else:
            pts = np.array(dataset[i]['pts'])
            class_map[label].append(pts)

    for i in class_map:
        data = np.array(class_map[i])
        mean = np.mean(data,axis = 0)
        mean_sample_dict[i] = mean
    # dataset.merge_dataset(dataset)
    with open('mean_sample_dict.pickle', 'wb') as handle:
                pickle.dump(mean_sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not test_loader :

        # # test label maps 
        # pprint(cfg_data.label_map_arr.tolist())
        # print(np.unique(cfg_data.label_map_arr))

        print("Number of samples = ", len(dataset))
        idx = random.randint(0, len(dataset)-1)
        data = dataset[idx]
        pts = data.pts
        label = data.label

        print('pts', pts.dtype, pts.shape, pts.max(), pts.min())
        print('label', label)

        sys.exit()

    # test loader
    from torch.utils.data import DataLoader as TorchDataLoader

    dataloader = TorchDataLoader(dataset, batch_size=4, shuffle=True)

    if not test_time :
        iter_loader = iter(dataloader)
        data = next(iter_loader)
        print(type(data))

        pts = data.pts
        label = data.label

        print('pts', pts.dtype, pts.shape, pts.max(), pts.min())
        print('label', label)

        sys.exit() 

    for data in tqdm(dataloader) :
        pts = data.pts
        label = data.label
        pts = pts.cuda(0, non_blocking=True)
        label = label.to(0, non_blocking=True)

    sys.exit()               

    
