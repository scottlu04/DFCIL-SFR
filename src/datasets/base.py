import sys
import os, os.path as osp 

from typing import List, Tuple, Sequence, Any, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from easydict import EasyDict as edict
from functools import partial

try :
    from .helpers import *
except :
    from helpers import *

sys.path.append('..')

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *


class Dataset(TorchDataset) :
    def __init__(self,
        mode: str,
        split_filepath: str,
        cfg: Config_Data,
        n_add_classes: int,
        n_known_classes: int,
        n_total_classes: int,
        rm_global_scale: bool,
        drop_seed: int,
        few_shot_seed: int =-1,
        few_shot_size: int =-1,
    ) -> None :

        dataset_l = ['hgr_shrec_2017',  'ego_gesture', 'ntu']

        assert mode in cfg.modes, f"Training mode {mode} must be one of {cfg.modes}"
        assert cfg.name in dataset_l, f"Dataset name ({cfg.name}) must be one of {dataset_l}"
        assert osp.isdir(cfg.root_dir), "Root directory not found {cfg.root_dir}"

        self.mode = mode
        self.split_filepath = split_filepath
        self.cfg = cfg
        
        # classes to keep
        if n_add_classes > 0 :
            self.keep_class_l = get_add_class_list(
                                    n_add_classes, 
                                    n_known_classes, 
                                    n_total_classes, 
                                    drop_seed,
            )
            # print(self.keep_class_l)

        else :
            self.keep_class_l = None
         
        self.full_file_list, self.file_list = get_file_list(
                            self.cfg.name, 
                            self.cfg.root_dir, 
                            self.split_filepath, 
                            self.mode,
                            self.keep_class_l,
        )
        if few_shot_size != -1 and mode =="train":
            self.few_shot_file_list = []
            categorized_data ={}
            for sample in self.file_list:
                class_label = sample[1]  
                if class_label not in categorized_data:
                    categorized_data[class_label] = []
                categorized_data[class_label].append(sample) 
            
            for label in categorized_data:
                temp = np.array(categorized_data[label])
                # row_indices = np.random.choice(temp.shape[0], sample_size_per_category, replace=False)
                # samples.append(temp[row_indices])
                idxs = np.random.RandomState(few_shot_seed).choice(temp.shape[0], few_shot_size, replace=False)
                self.few_shot_file_list.append(temp[idxs])
            self.few_shot_file_list = np.concatenate(self.few_shot_file_list)
  
            self.file_list = self.few_shot_file_list
        self.loader_pts = partial(
                            globals()['read_pts_' + cfg.name], 
                            rm_global_scale=rm_global_scale,
        )

        self.n_classes = None


    def __len__(self) :
        return len(self.file_list)


    def read_pts(self, fpath: str) -> np.ndarray :
        return self.loader_pts(fpath)

    
    def merge_dataset(self, other) :
        assert type(self) == type(other), \
            f"Type of this dataset {type(self)} does not match the other {type(other)}."
        
        print(f"(Before merging) Number of files = {len(self)}")
        print(f"(Before merging) Number of classes = {len(self.keep_class_l)}")

        self.file_list.extend(other.file_list)
        if self.keep_class_l is None :
            self.keep_class_l = other.keep_class_l
        else :
            self.keep_class_l = list( set(self.keep_class_l).union(other.keep_class_l) )
        if self.keep_class_l is not None :
            self.keep_class_l.sort()

        print(f"(After merging) Number of files = {len(self)}")
        print(f"(After merging) Number of classes = {len(self.keep_class_l)}")


    # naive coreset appending
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


          