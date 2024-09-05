import sys
import os, os.path as osp
import pickle
from easydict import EasyDict as edict
from typing import Any, Callable, Sequence, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

try :
    from .base import Dataset as BaseDataset
    from .helpers import *
    from . import transforms
    from .tool import *
except :
    from base import Dataset as BaseDataset
    from helpers import *
    import transforms
    from tool import *

sys.path.append('..');

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *



class Dataset(TorchDataset) :
    def __init__(self,
        mode: str,
        split_type: str,
        n_views: int,
        cfg: Config_Data,
        cfg_xforms: dict, 
        n_add_classes: int = -1,
        n_known_classes: int = 0,
        rm_global_scale: bool = False,
        drop_seed: int = -1,        
        few_shot_seed: int = -1,
        few_shot_size: int = -1,
    ) -> None :
        
        if few_shot_seed != -1:
            raise NotImplementedError
        cfg_xforms = edict(cfg_xforms);
        self.xforms = transforms.get_transforms_from_cfg_ntu(cfg_xforms);

        # self.window_size= cfg_xforms.stratified_sample.n_samples
        # self.p_interval= [0.5,1]
        #print(len(self.p_interval))
        self.to_tensor = transforms.ToTensor();
        #cross-subject
        self.n_views = n_views
        n_total_classes = cfg.get_n_classes(split_type)
        if n_add_classes > 0 :
            self.keep_class_l = get_add_class_list(
                                    n_add_classes, 
                                    n_known_classes, 
                                    n_total_classes, 
                                    drop_seed,
            )
        else :
            self.keep_class_l = None

        self.mode = mode
        file_path = '/media/exx/HDD/zhenyulu/ntu/ntu60_3danno.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        split, data = data['split'], data['annotations']
        identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        
        if self.mode == 'train':
            split = set(split["xsub_train"])
            self.data = data = [x for x in data if x[identifier] in split]
            # self.label = np.argmax(data['y_train'], axis=-1)
        elif self.mode == 'test' or 'val':  
            split = set(split["xsub_val"])
            self.data = data = [x for x in data if x[identifier] in split]
        else:
            raise NotImplementedError('data split only supports train/test')


        
        # TODO fix class incremental setting
        locs = []
        #print(locs)
        new_data = []
        for sample in self.data:
            if sample['label'] in self.keep_class_l:
                new_data.append(sample)

        self.data = new_data
        # for k in self.keep_class_l:
        #     loc = (self.label == k).nonzero()[0]
        #     #print(loc)
        #     locs.append(loc)
        #     #locs = np.concatenate((locs,loc))
        # locs = np.concatenate((locs))
        # #print(locs)
        # self.label = self.label[locs]
        # self.sample = self.data[locs]


    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        # pts = self.data[idx]['keypoint']
        # label = self.data["label"]
        # data = 
        # print(self.data[idx])
        temp = self.xforms(self.data[idx])
        pts_l = [];
        for _ in range(self.n_views) :
            pts = temp['keypoint']
            pts = torch.squeeze(self.to_tensor(pts))
            pts_l.append(pts)
        
        pts = torch.stack(pts_l, dim=0);
        label = temp["label"]
        
        data = edict({
            'pts': pts,
            'label': label,
        });
        return data; 

    def append_coreset(self, coreset, ic, only=False):
        len_core = len(coreset)
        if (self.mode == 'train' or self.mode == 'val') and (len_core > 0):
            if only:
                self.file_list = coreset
            else:
                len_data = len(self.file_list)
                sample_ind = np.random.choice(len_core, len_data)
                if ic:
                    self.file_list.extend([coreset[i] for i in range(len(coreset))])
                else:
                    self.file_list.extend([coreset[i] for i in sample_ind])

    # naive coreset update
    def update_coreset(self, coreset, coreset_size, seen, class_mapping):
        #coreset_size = 200
        state = np.random.get_state()
        np.random.seed(1994)
        #print(class_mapping)
        mapped_targets = [class_mapping[str(self.file_list[i][1])] for i in range(len(self.file_list))]
        #print(self.file_list)
        #print(mapped_targets)
        #print(seen)
        for k in seen:
            locs = (mapped_targets == k).nonzero()[0]
            #print(locs)
            num_data_k = 10# max(1, round(coreset_size * len(locs) / 100))
            #print(num_data_k)
            #locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            locs_chosen = np.random.permutation(locs)[:num_data_k]
            coreset.extend([self.file_list[loc] for loc in locs_chosen])
        np.random.set_state(state)
        print(coreset)
        print(len(coreset))
        return coreset

if __name__ == "__main__" :

    test_loader = True;
    test_time = True;


    import yaml, random
    from pprint import pprint
    from tqdm import tqdm
    from configs.datasets import ntu
    from utils.colors import *

    root_dir = '/media/exx/HDD/zhenyulu/ntu';
    # root_dir = '/data/ashubhra/agr/cvpr_2023/hgr_shrec_2017/vanilla/supcon/finetune/initial_1k/drop/class_6_11_4_10_2_8/inverted_samples';
    cfg_file = '../configs/params/ntu/Base.yaml';
  
    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

    # mode = 'train';
    # mode = 'val';
    mode = 'train';
    split_type ='cs'
    cfg_data = ntu.Config_Data(root_dir)
    dataset = Dataset(
                mode, 
                split_type,
                cfg_data,
                cfg_params.transforms[mode],
                n_add_classes = 12,
                drop_seed=1

    );
   
    # # test loader
    from torch.utils.data import DataLoader as TorchDataLoader

    dataloader = TorchDataLoader(dataset, batch_size=20, shuffle=True);



    for data in tqdm(dataloader) :
        pts = data.pts;
        label = data.label;
        pts = pts.cuda(0, non_blocking=True);
        label = label.to(0, non_blocking=True);
        print(pts.size())

    # sys.exit();               

    
