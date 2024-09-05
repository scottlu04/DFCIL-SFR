import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchmetrics
import sys
import os.path as osp
import os
import copy
import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
from sklearn.decomposition import TruncatedSVD

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils 


##########################################
#            TEACHER CLASSES             #
##########################################

class Teacher_v4(nn.Module):

    def __init__(self, solver, classifier, sample_shape, iters, class_idx,num_inverted_class,num_known_classes, deep_inv_params, train = True, config=None, step = 0):

        super().__init__()
        self.solver = solver
        self.solver.eval()

        self.classifier = classifier
        # self.classifer.eval()
        self.sample_shape = sample_shape
        self.iters = iters
        self.config = config
        self.step = step
        self.few_shot_feat_aug = config.few_shot_feat_aug

        # hyperparameters
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        self.inv_mean = deep_inv_params[5]
        self.inv_std = deep_inv_params[6]
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_known_classes = num_known_classes
        self.num_inverted_class = num_inverted_class
        self.num_k =  self.num_inverted_class + self.num_known_classes
        
        #synthetic feature size
        self.class_size = config.increm.learner.n_samples_per_class

        self.target_features = {}
        self.targets = {}
        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()




    # use saved 
    def generate_target_feature(self, features_dir):
        fname = "sampled_features" +str(self.step)+ '.pkl'
        fpath = osp.join(features_dir, fname)
        if not osp.exists(fpath):
            fname = "features" + '.pkl'
            fpath = osp.join(features_dir, fname)
            data  = utils.load_pickle(fpath);
    
        
            pre_sample_class_size =  self.class_size *5
            proto_mean, proto_var,reduced_mean,reduced_var, feat_dict, reduced_feat_dict, directions_dict,u_dict, svd_learner = get_prototype(data, self.class_size, 6)
            
            # save reduced featurs used for visualizaiton and confirmation >>>>>>>>>>>>>>>>>>>>
            fname = "reduced_features" +str(self.step-1) +'.pkl'
            fpath = osp.join(features_dir, fname)
            
            info_dict = {
                'reducd_features': reduced_feat_dict,
                'feat_dict':feat_dict,
                'u_dict': u_dict,
                'proto_mean': proto_mean,
                'proto_var': proto_var,
                'reduced_mean': reduced_mean,
                'reduced_var': reduced_var,
                'directions_dict':directions_dict,
                'svd_learner':svd_learner,
            }
            
            fpath_full = osp.join(features_dir,'info_dict.pkl')
            if osp.exists(fpath_full):
                info_dict_old  = utils.load_pickle(fpath_full);
                print(info_dict_old['proto_mean'].keys())
                info_dict_full  = copy.deepcopy(info_dict_old);
                for name in info_dict:    
                    info_dict_full[name].update(info_dict[name])

                print(info_dict_old['proto_mean'].keys())
                utils.save_pickle(fpath_full, info_dict_full);
            else:
                utils.save_pickle(fpath_full, info_dict);
            
            
            self.target_features, self.targets = sampling(info_dict, self.class_size, pre_sample_class_size, 'random_full_dim')
            def mahalanobis_distance(x, mean, covariance):
                """Compute Mahalanobis distance between a point x and a multivariate Gaussian distribution
                characterized by mean and covariance."""

                cov_inv = np.linalg.inv(covariance)
                diff = x - mean
                md = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))
                return md
            saved = 1
            if self.few_shot_feat_aug and self.step > 0 and saved == 1:

                d_maha_threshold = self.config.d_maha_threshold
                rej_dict = {}
                # outer 
                for new_id in self.target_features:
                    while (1):
                        rej_dict[new_id] = set()
                        # count = 0
                        for idx, feat in enumerate(self.target_features[new_id]):
                            # count = 0
                            # compared with old class
                            for old_id in info_dict_old['proto_mean']:        
                                # count +=1
                                mean = info_dict_old['proto_mean'][old_id][0]
                                cov = info_dict_old ['proto_var'][old_id]
                                cov =np.diag(np.diag(cov))
                                val = mahalanobis_distance(feat,mean,cov)
                                if not np.isnan(val):
                                    if val < d_maha_threshold:
                                        rej_dict[new_id].add(idx)
                        
                            #comapred with other new classes
                            for new_other_id in info_dict['proto_mean']:        
                                # count +=1
                                # print('compare')
                                # print(new_other_id)
                                # print(new_id)
                                if new_id == new_other_id:
                                    continue 
                                mean = info_dict['proto_mean'][new_other_id][0]
                                cov = info_dict ['proto_var'][new_other_id]
                                cov =np.diag(np.diag(cov))
                                val = mahalanobis_distance(feat,mean,cov)
                                if not np.isnan(val):
                                    if val < d_maha_threshold:
                                        rej_dict[new_id].add(idx)
                            
                                    # print(old_id, val)
                        filtered_size = pre_sample_class_size - len(rej_dict[new_id])
                        print(new_id,'len after filter',filtered_size)
                        if filtered_size >  self.class_size:
                            break
                        else:
                            d_maha_threshold -= 5

            print('sanity check')
            for id in reduced_var:
                # if self.few_shot_feat_aug and self.step > 0:
                #     if saved == 1:
                #         aug_feature  = utils.load_pickle(osp.join(features_dir, 'aug_features.pkl'))
                #         self.target_features = aug_feature[0]
                #         self.target = aug_feature[1]
                #     else:
                #         self.target_features[id] =np.delete(self.target_features[id], list(rej_dict[id]), axis=0)  #target_features[id][accpet_dict[id],:]

                #         idxs = np.random.RandomState(1).choice(self.target_features[id].shape[0], self.class_size, replace=False)
                #         self.target_features[id] = self.target_features[id][idxs]
                #         self.targets[id] = self.targets[id][:self.class_size]
                # else:
                    # has classifier
                    acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes = self.config.num_total_classes).cuda()
                    acc_meter.reset()
                    
                    outputs = self.classifier(torch.tensor(self.target_features[id]).to(torch.float).cuda())
                    targets = torch.full((pre_sample_class_size,), id).cuda()
                    acc = acc_meter(outputs,targets) * 100
                    print('feed sampled_features to classifier',id, acc)
            
                    predicted_labels = torch.argmax(outputs, dim=1)
                    correct_indices = (predicted_labels.cpu() == torch.full((pre_sample_class_size,), id)).nonzero().squeeze()
                    # print(correct_indices)
                    self.target_features[id] = self.target_features[id][correct_indices] 
                    # print(self.target_features[id].shape)
                    idxs = np.random.RandomState(1).choice(self.target_features[id].shape[0], self.class_size, replace=False)
                    self.target_features[id] = self.target_features[id][idxs]
                    self.targets[id] = self.targets[id][idxs]
                        # print(self.target_features[id].shape)

            self.targets_list = []
            self.features_list =[]
            self.inverted_sample_from_mean = []

            for id in self.targets:
                self.targets_list.append(self.targets[id])
                self.features_list.append(self.target_features[id])
            self.targets_list = np.concatenate(self.targets_list)  
            print('#################### aug target ')
            print( self.targets_list)
            self.features_list =  np.concatenate(self.features_list)  
            
            # save featurs used for inversion >>>>>>>>>>>>>>>>>>>>
            print(self.features_list.shape)
            fname = "sampled_features" + str(self.step) + '.pkl'
            fpath = osp.join(features_dir, fname)
            sampled_features = {
                'features': self.features_list,
                'targets': self.targets_list,
            }
            utils.save_pickle(fpath, sampled_features);
        else:
            saved_data  = utils.load_pickle(fpath);
            self.targets_list = saved_data['targets']
            self.features_list =saved_data['features']






    def generate_feature_space(self, class_size, inversion_batch_size, dataloader_batch_size, log_dir_task, inverted_sample_dir):
        # self.class_size = 100
        saved = False
        fname = "saved_inverted_sample_" +str(self.step)+ '.pkl'
        fpath = osp.join(inverted_sample_dir, fname)
        if not osp.exists(fpath):
            self.inverted_pts = self.features_list
            self.inverted_targets = self.targets_list
            self.inverted_outputs = self.targets_list
            fname = "saved_inverted_sample_" +str(self.step-1)+ '.pkl'
            fpath = osp.join(inverted_sample_dir, fname)
            if self.num_known_classes !=0:
                print("load saved data")
                saved_data  = utils.load_pickle(fpath)
                # self.inverted_pts.extend(saved_data["pts"])
                # self.inverted_targets.extend(saved_data["target"])
                # self.inverted_outputs.extend(saved_data["output"])
                self.inverted_pts = np.concatenate([self.inverted_pts,saved_data["pts"]])
                self.inverted_targets = np.concatenate([self.inverted_targets,saved_data["target"]])
                self.inverted_outputs =  np.concatenate([self.inverted_targets,saved_data["target"]])
                # os.remove(fpath)
                # print(self.inverted_targets)
            save_dict = {
                'pts': self.inverted_pts,
                'target': self.inverted_targets,
                'output': self.inverted_outputs,
            }

            #new name
            fname = "saved_inverted_sample_" +str(self.step)+ '.pkl'
            fpath = osp.join(inverted_sample_dir, fname)
            utils.save_pickle(fpath, save_dict);
        else:
            fname = "saved_inverted_sample_" +str(self.step)+ '.pkl'
            fpath = osp.join(inverted_sample_dir, fname)

            saved_data  = utils.load_pickle(fpath);
            self.inverted_pts = saved_data["pts"]
            self.inverted_targets = saved_data["target"]
            self.inverted_outputs = saved_data["output"]


        print(self.inverted_pts[0].shape)
        # Create a new dataset with the inverted samples
        self.inverted_dataset = InvertedDataset(self.inverted_pts, self.inverted_targets, self.inverted_outputs)
        # Create a new dataloader with the inverted dataset
        self.inverted_dataloader = DataLoader(self.inverted_dataset, batch_size=dataloader_batch_size, shuffle=True, num_workers=self.config.workers, pin_memory=True)
        # Create a iterator for the inverted dataloader
        self.inverted_iterator = iter(self.inverted_dataloader)


        
    

    def sample(self):
        # Return a batch of inverted samples
        try:
            inputs, targets, outputs = next(self.inverted_iterator)
        except StopIteration:
            self.inverted_iterator = iter(self.inverted_dataloader)
            inputs, targets, outputs = next(self.inverted_iterator)
        return inputs, targets, outputs

    

    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]


        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
             feature, feature_cl = self.solver.forward_feature(x)

        return feature, feature_cl
    def generate_scores_pen_ib(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            features,z = self.solver.forward_feature(x)

        return z





def get_mean_cov(data):
        label_list,feat_list,target_list = data 
        proto_mean, proto_var, samples_per_class,feat_dict = {}, {}, {},{};
        count_reject = 0;
        for k in range(feat_list.shape[0]):
            feat = feat_list[k];
            # print(feat.shape)
            label = label_list[k];
            target = target_list[k];
            dim, dtype = feat.size, feat.dtype;
            # init class-wise mean
            if label != target : # reject the sample
                # print("reject")  
                count_reject += 1;
                continue;
            if label not in proto_mean:
                feat_dict[label] = []
                proto_mean[label] = np.zeros((1, dim), dtype=dtype);
                samples_per_class[label] = 0;
            
            feat_dict[label].append(feat)
            proto_mean[label] += feat[None, :];
            samples_per_class[label] += 1;
        for label in samples_per_class :
            proto_mean[label] /= samples_per_class[label];
        print("samples rejected",count_reject)
        
        for c in feat_dict:
            feat_dict[c] = np.stack(feat_dict[c])
        # ============= compute global variance ================ #
        count_reject = 0;
        for k in range(feat_list.shape[0]) :
            feat = feat_list[k];
            
            label = label_list[k];
            target = target_list[k];
            dim, dtype = feat.size, feat.dtype;
            # init class-wise mean
            if label != target : # reject the sample
                # print("reject")  
                count_reject += 1;
                continue;
            if label not in proto_var :
                proto_var[label] = np.zeros((dim, dim), dtype=dtype);

            feat = feat[None, :] - proto_mean[label];
            proto_var[label] += (feat.T @ feat);
        # normalize 
        for label in samples_per_class :
            proto_var[label] /= samples_per_class[label];
        print("samples rejected",count_reject)
        return proto_mean, proto_var, feat_dict

def get_mean_cov_no_filter(data): #does not reject samples that are misclassified
        feat_list,target_list = data 
        proto_mean, proto_var, samples_per_class,feat_dict = {}, {}, {},{};
        for k in range(feat_list.shape[0]):
            feat = feat_list[k];
            label = target_list[k];
            # print(label)
            dim, dtype = feat.size, feat.dtype;
            # init class-wise mean
            if label not in proto_mean:
                feat_dict[label] = []
                proto_mean[label] = np.zeros((1, dim), dtype=dtype);
                samples_per_class[label] = 0;
            
            feat_dict[label].append(feat)
            proto_mean[label] += feat[None, :];
            samples_per_class[label] += 1;
        for label in samples_per_class :
            proto_mean[label] /= samples_per_class[label];
        
        for c in feat_dict:
            feat_dict[c] = np.stack(feat_dict[c])
        # ============= compute global variance ================ #
        for k in range(feat_list.shape[0]) :
            feat = feat_list[k];

            label = target_list[k];
            dim, dtype = feat.size, feat.dtype;
            # init class-wise mean
            if label not in proto_var :
                proto_var[label] = np.zeros((dim, dim), dtype=dtype);

            feat = feat[None, :] - proto_mean[label];
            proto_var[label] += (feat.T @ feat);
        # normalize 
        for label in samples_per_class :
            proto_var[label] /= samples_per_class[label];
        return proto_mean, proto_var, feat_dict

def get_prototype(data, class_size, pca_param = 6 ,no_filter = 0) :
    
        if no_filter:
            proto_mean, proto_var, feat_dict = get_mean_cov_no_filter(data)
        else:
            proto_mean, proto_var, feat_dict = get_mean_cov(data)
        
        #eig_decomposition 
        reduced_var = {}
        reduced_mean = {}
        reduced_feat_dict ={}
        directions_dict =  {}
        u_dict = {}
        svd_learner = {}
        for c in proto_mean:
            svd_learner[c] =(TruncatedSVD(n_components=pca_param))
            reduced_feat_dict[c] = svd_learner[c].fit_transform(feat_dict[c])
            # n_keep = 3
            # print(c, reduced_feat_dict[c].shape)
            reduced_var[c] = np.cov(reduced_feat_dict[c].T)
            reduced_mean[c] = np.mean(reduced_feat_dict[c].T,axis = 1)
            d, n = reduced_feat_dict[c].shape;
            # print(c, reduced_var[c].shape)
            directions = np.diag( [1]*n)
            principal_axes = utils.get_mixture_of_samples(directions, n, n, class_size, 4);
            # print(principal_axes)
            directions_dict[c] = principal_axes
            u_dict[c] = None
            # svd_learner
            # print(reduced_mean[c].shape)    
        
        return proto_mean, proto_var,reduced_mean,reduced_var,feat_dict, reduced_feat_dict, directions_dict,u_dict, svd_learner

def get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        d = torch.dist(pred, target) / norm_;
        return d;

def sampling(reduced_features,class_size,pre_sample_class_size, sampling_method ='random_full_dim'):
        reduced_features_sampled = {}
        target_features ={}
        targets = {}
        sampling_method = 'random_full_dim'
        for id in reduced_features['reduced_var']:
            if sampling_method == 'wts':
                target_features[id],targets[id], reduced_features_sampled[id] = wts(
                    reduced_features['reduced_mean'][id],
                    reduced_features['proto_mean'][id],
                    reduced_features['reduced_var'][id],
                    reduced_features['directions_dict'][id],
                    reduced_features['u_dict'][id],
                    id,
                    reduced_features['reduced_feat_dict'][id],
                    reduced_features['svd_learner'][id])
            elif sampling_method == 'random_full_dim':
                print('using random sampling from full-dimension')
                #sampling from full-dimension
                features_list = []
                targets[id] = np.array([id]*pre_sample_class_size)
                # random_sample.append(np.random.multivariate_normal(reduced_features['proto_mean'][i][0], reduced_features['proto_var'][i],300))
                target_features[id] = np.random.multivariate_normal(reduced_features['proto_mean'][id][0], reduced_features['proto_var'][id],pre_sample_class_size)
            elif sampling_method == 'random_reduced_dim':
                print('using random sampling from reduced-dimension')
                target_features[id] = random_sampling_reduced(
                    reduced_features['reduced_mean'][id],
                    reduced_features['reduced_var'][id],
                    class_size,
                    id,
                    reduced_features['svd_learner'][id])
                targets[id] = np.array([id] * pre_sample_class_size)
                
        return target_features,targets

def random_sampling_reduced(mean, var, class_size, class_id, svd_learner):
  
        # print('class_id',class_id)
        dtype = torch.get_default_dtype();
        reduced = np.random.multivariate_normal(mean, var,class_size)
        features_list = svd_learner.inverse_transform(reduced)

        return features_list



class InvertedDataset(Dataset):
    def __init__(self, pts, targets, outputs):
        self.pts = pts
        self.targets = targets
        self.outputs = outputs

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        x_i = self.pts[idx]
        y = np.array(self.targets[idx])
        #TODO change back 
        y_hat = np.array(self.targets[idx])
        # y_hat = self.outputs[idx]

        # Convert to torch tensors
        x_i = torch.from_numpy(x_i)
        y = torch.from_numpy(y)
        
        y_hat = torch.from_numpy(y_hat)

        return x_i, y, y_hat
    