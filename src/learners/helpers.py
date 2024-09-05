import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchmetrics
import sys
import os.path as osp
import os
import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils 


##########################################
#            TEACHER CLASSES             #
##########################################

class Teacher_v1(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat



class Teacher_v2(nn.Module):

    def __init__(self, solver, sample_shape, iters, class_idx,num_inverted_class,num_known_classes, deep_inv_params, train = True, config=None):

        super().__init__()
        self.solver = solver
        # self.generator = generator
        # self.gen_opt = gen_opt
        self.solver.eval()
        self.sample_shape = sample_shape
        self.iters = iters
        self.config = config

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

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers


    def generate_inverted_samples(self, class_size, inversion_batch_size, dataloader_batch_size, log_dir_task, inverted_sample_dir):

        # Calculate number of batches
        size = class_size * self.num_inverted_class
        num_batches = math.ceil(size / inversion_batch_size)
        num_samples_batch = inversion_batch_size
        targets_list = [i for i in range(self.num_known_classes,self.num_known_classes + self.num_inverted_class)] * class_size
        targets_list = np.array(targets_list)
        #print(range(self.num_known_classes,self.num_known_classes + self.num_inverted_class))
        #print(targets_list)
        # clear cuda cache
        torch.cuda.empty_cache()
        self.solver.eval()
        self.original_pts = []
        self.inverted_pts = []
        self.inverted_targets = []
        self.inverted_outputs = []

        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes= self.config.num_total_classes).cuda()
        tb_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'inverted_samples'))
        print(f"Generating {size} inverted samples....................")
        bar = tqdm(total=num_batches, leave=True, desc='Inversion steps', dynamic_ncols=False)
        steps = 0
        for batch_id in range(num_batches):
            if batch_id == num_batches - 1:
                num_samples_batch = size - (batch_id * num_samples_batch)
            inputs = torch.normal(self.inv_mean ,self.inv_std , size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
            inputs = nn.Parameter(inputs)
            optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.9, 0.999], eps = 1e-8)
            self.original_pts.extend(inputs.data.cpu().numpy().copy())
            targets = torch.LongTensor(targets_list[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]).cuda()
            #print(f"Input stats: mean: {inputs.mean()} -- std: {inputs.std()}")
            acc_meter.reset()
            for iteration in range(self.iters):
                # forward with images
                self.solver.zero_grad()

                # content
                outputs = self.solver(inputs)
                loss = self.criterion(outputs / self.content_temp, targets) * self.content_weight

                # R_feature loss
                for mod in self.loss_r_feature_layers: 
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    loss = loss + loss_distr

                optimizer.zero_grad()
                # backward pass
                loss.backward()
                with torch.no_grad():
                    # Update input image
                    optimizer.step() 
                    acc = acc_meter(outputs, targets) * 100

                if acc.item() > self.config.increm.learner.inverted_acc_thred:
                    break 

            # update tb_logger
            tb_logger.update({
                'inverted_training_loss': loss.item(),
                }, step=steps, prefix="loss" )  
            tb_logger.update({
                'inverted_training_acc': acc.item(),
                }, step=steps, prefix="acc" ) 

            steps += 1
            # update progress bar
            bar.update()
            # clear cuda cache
            torch.cuda.empty_cache()
            #print(f"Inverted stats: mean: {inputs.mean()} -- std: {inputs.std()}")
            self.inverted_pts.extend(inputs.data.cpu().numpy())
            self.inverted_targets.extend(targets.cpu().numpy())

        bar.close() 
        tb_logger.close()  
        acc_meter.reset()

        print("Check accuracy with the inverted samples:")

        # Inverted input accuracy
        num_samples_batch = inversion_batch_size
        for batch_id in range(num_batches):
            if batch_id == num_batches - 1:
                num_samples_batch = size - (batch_id * num_samples_batch)
            inputs_list = self.inverted_pts[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            targets_list = self.inverted_targets[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            inputs = torch.zeros(size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
            targets = torch.zeros(size=(num_samples_batch,)).cuda()
            for i in range(len(inputs_list)):
                inputs[i] = torch.from_numpy(inputs_list[i])
                targets[i] = targets_list[i]  
            with torch.inference_mode():
                outputs = self.solver(inputs)
                acc = acc_meter(outputs, targets) * 100
                self.inverted_outputs.extend(outputs.cpu().numpy())
        print(f"Acc inverted samples = {acc_meter.compute() * 100}")
        acc_meter.reset()

        #save inverted samples
        fname = "saved_inverted_sample" + '.pkl'
        fpath = osp.join(inverted_sample_dir, fname)
        if self.num_known_classes !=0:
            print("load saved data")
            saved_data  = utils.load_pickle(fpath);
            self.inverted_pts.extend(saved_data["pts"])
            self.inverted_targets.extend(saved_data["target"])
            self.inverted_outputs.extend(saved_data["output"])
            os.remove(fpath)

        save_dict = {
            'pts': self.inverted_pts,
            'target': self.inverted_targets,
            'output': self.inverted_outputs,
        }
        utils.save_pickle(fpath, save_dict);

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
             y_hat = self.solver.forward_feature(x)

        return y_hat


class Teacher_v3(nn.Module):

    def __init__(self, solver, sample_shape, iters, class_idx,num_inverted_class,num_known_classes, deep_inv_params, train = True, config=None, step = 0):

        super().__init__()
        self.solver = solver
        self.solver.eval()
        self.sample_shape = sample_shape
        self.iters = iters
        self.config = config
        self.step = step

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


        self.target_features = {}
        self.targets = {}
        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers

    def generate_target_feature(self, features_dir, class_size):
        fname = "sampled_features" +str(self.step)+ '.pkl'
        fpath = osp.join(features_dir, fname)
        if not osp.exists(fpath):
            fname = "features" + '.pkl'
            fpath = osp.join(features_dir, fname)
            data  = utils.load_pickle(fpath);
            class_size = 300
            self.class_size = class_size
            proto_mean, proto_var,reduced_mean,reduced_var,feat_dict, reduced_feat_dict, directions_dict,u_dict = get_prototype(data, class_size, 0.90)
            
            # save reduced featurs used for visualizaiton and confirmation >>>>>>>>>>>>>>>>>>>>
            fname = "reduced_features" +str(self.step-1) +'.pkl'
            fpath = osp.join(features_dir, fname)
            reduced_features = {
                'features': reduced_feat_dict,
                'u': u_dict,
                'proto_mean': proto_mean,
                'proto_var': proto_var,
                'reduced_mean': reduced_mean,
                'reduced_var': reduced_var,

            }
            utils.save_pickle(fpath, reduced_features);
            # print(diff_entropy2(reduced_var[0]))
            # print(diff_entropy(reduced_var[0]))
            reduced_features_sampled = {}
            for id in reduced_var:
                self.target_features[id],self.targets[id], reduced_features_sampled[id]  = self.wts(reduced_mean[id],proto_mean[id],reduced_var[id],directions_dict[id],u_dict[id],id,reduced_feat_dict[id])
            # print(target_features[id].shape)
                print(reduced_features_sampled[id].shape)
            self.targets_list = []
            self.features_list =[]
            reduced_feature_list  = [] # sampled feature in reduced dimension for visualization 
            for id in self.targets:
                self.targets_list.append(self.targets[id])
                self.features_list.append(self.target_features[id])
                reduced_feature_list.append(reduced_features_sampled[id])
            self.targets_list = np.concatenate(self.targets_list)  
            self.features_list =  np.concatenate(self.features_list)  
            # reduced_feature_list = np.concatenate(reduced_feature_list) 
            # save featurs used for inversion >>>>>>>>>>>>>>>>>>>>
            print(self.features_list.shape)
            fname = "sampled_features" + str(self.step) + '.pkl'
            fpath = osp.join(features_dir, fname)
            sampled_features = {
                'features': self.features_list,
                'targets': self.targets_list,
                'features_reduced': reduced_feature_list
            }
            utils.save_pickle(fpath, sampled_features);
        else:
            saved_data  = utils.load_pickle(fpath);
            self.targets_list = saved_data['targets']
            self.features_list =saved_data['features']
    # def random_sampling(self, mean, var, class_id):
    #     features_list = []
    #     targets_list =[]

    def ground_truth(self, data):
        features_list = []
        targets_list =[]

    def wts(self, mean, real_mean, var, directions, U, class_id,reduced_features):
        dif_entropy = diff_entropy2(reduced_features,100) 
        # dif_entropy = diff_entropy(var) 
        print(dif_entropy)
        print('class_id',class_id)
        features_list = []
        targets_list =[]
        picked_point = None
        dtype = torch.get_default_dtype();
                
        mean = torch.from_numpy(mean).to(dtype)
        # p_ax_t = torch.from_numpy(p_ax).to(dtype)
        j = 0
        for d in directions:
            d = torch.from_numpy(d).to(dtype)
            # print(j)
            # j +=1
            lowest = 1000
            alpha  = 0.001
            # shift_ = alpha * d;
            new_point = mean.clone();
            # print(">>>>>>>>>>>>>>>>>>>>>>")
            distance  = 0
            for i in range(1000):

                # new_point = mean + 0.01 * d * i
                px = multivariate_normal.pdf(new_point, mean=mean, cov=var)
                loss = abs(-np.log(px)-dif_entropy)
                alpha = 0.001 * (np.log(px)+dif_entropy)
                new_point.add_(alpha * d);
                distance += alpha
                # print(i,-np.log(px)*px)
                # print(loss,alpha,-np.log(px))
                if loss < lowest:
                    lowest = loss
                    picked_point = new_point.cpu().clone()
                    if lowest <0.01:
                        print(loss,-np.log(px), distance)
                        break
                    # print(lowest)
                    
            if lowest >0.5:
                print(lowest)
                print('error')
            else:
                features_list.append(picked_point)
                targets_list.append(class_id)
        
        features_list = np.stack(features_list)
        targets_list = np.array(targets_list)
        # print(features_list.shape)
        real_mean = np.squeeze(real_mean)
        # print(real_mean.shape) 
        expanded_mean = np.repeat(real_mean[ np.newaxis,:], features_list.shape[0], axis=0)
        # print(expanded_mean.shape)
        # print(features_list.shape)
        reduced_feature = features_list
        features_list = features_list.dot(U.T) + expanded_mean
        # print(np.linalg.inv(U).shape)
        # features_list = np.matmul(features_list, np.linalg.inv(U))

        return features_list, targets_list, reduced_feature

    def generate_inverted_samples(self, class_size, inversion_batch_size, dataloader_batch_size, log_dir_task, inverted_sample_dir):
        self.class_size = 100
        saved = False
        fname = "saved_inverted_sample_" +str(self.step)+ '.pkl'
        fpath = osp.join(inverted_sample_dir, fname)
        if not osp.exists(fpath):
            # Calculate number of batches
            size = self.class_size * self.num_inverted_class
            num_batches = math.ceil(size / inversion_batch_size)
            num_samples_batch = 1# inversion_batch_size
            # targets_list = [i for i in range(self.num_known_classes,self.num_known_classes + self.num_inverted_class)] * class_size
            # targets_list = np.array(targets_list)
            # targets_list = []
            # features_list =[]
            # for id in self.targets:
            #     targets_list.append(self.targets[id])
            #     features_list.append(self.target_features[id])
            # # print(self.targets[id])
            # targets_list = np.concatenate(targets_list)  
            # features_list =  np.concatenate(features_list)  

            # save featurs used for inversion >>>>>>>>>>>>>>>>>>>>
            # print(features_list.shape)
            # fname = "sampled_features" + '.pkl'
            # fpath = osp.join(inverted_sample_dir, fname)
            # sampled_features = {
            #     'features': features_list,
            #     'targets': targets_list,
            # }
            # utils.save_pickle(fpath, sampled_features);



            # load mean sample for pior regua >>>>>>>>>>>>>>>>>>>>
            # fname = "mean_sample_dict" + '.pickle'
            # fpath = osp.join(inverted_sample_dir, fname)
            # mean_sample_dict  = utils.load_pickle(fpath);

            #print(range(self.num_known_classes,self.num_known_classes + self.num_inverted_class))
            #print(targets_list)
            # clear cuda cache
            torch.cuda.empty_cache()
            self.solver.eval()
            self.original_pts = []
            self.inverted_pts = []
            self.inverted_targets = []
            self.inverted_outputs = []

            acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes = self.config.num_total_classes).cuda()
            print(f"Generating {size} inverted samples....................")
            bar = tqdm(total=size, leave=True, desc='Inversion steps', dynamic_ncols=False)
            steps = 0
            # self.tolerance = 0.06
            self.tolerance = 0.3
            for idx in range(size):
                inputs = torch.normal(self.inv_mean ,self.inv_std , size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
                inputs = nn.Parameter(inputs)
                optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.9, 0.999], eps = 1e-8)
                self.original_pts.append(inputs.data.cpu().numpy().copy())
                
                targets = torch.tensor(self.targets_list[idx]).cuda()
                
                targets = targets[None]
                gt_feat = torch.tensor(self.features_list[idx]).cuda()
                gt_feat = gt_feat[None]
                # print(gt_feat.shape)
                # print(targets.shape)
                #print(f"Input stats: mean: {inputs.mean()} -- std: {inputs.std()}")
                acc_meter.reset()
                for iteration in range(self.iters):
                    # forward with images
                    self.solver.zero_grad()
                    # content
                    # outputs,feat = self.solver.forward_feature(inputs)
                    if self.config.model.name == 'dg_sta_var':
                        outputs,feat, mu, std = self.solver(inputs)
                    else:
                        outputs,feat, feature_cl= self.solver.forward_feature(inputs)
                    # if iteration == 0:
                    #     print(outputs)
                    #     print(targets)
                    # print(outputs.shape)
                    # print(feat.shape)
                    
                    #loss = self.criterion(outputs / self.content_temp, targets) * self.content_weight
                    i_loss = get_norm_dist(feat, gt_feat) # inversion loss
                
                    # mean_sample = torch.tensor(mean_sample_dict[targets_list[idx]]).cuda()
                    # mean_sample = mean_sample[None, ...]
                    l1 = nn.L1Loss(reduction='sum')
                    p_loss = 0  #l1(inputs, mean_sample)# prior loss
                    loss = i_loss #0.1* p_loss
                    # print(iteration,i_loss,p_loss)
                    optimizer.zero_grad()
                    # backward pass
                    loss.backward()
                    with torch.no_grad():
                        # Update input image
                        optimizer.step() 
                        acc = acc_meter(outputs, targets) * 100
                        # print(outputs, )
                        # print('iter',iteration,acc)
                    if (acc.item() > 99 and iteration >= 1000):#or (i_loss < self.tolerance and p_loss < 150):
                        print('success',idx, iteration,i_loss,p_loss)
                        # print(acc)
                        break 
                    # else:
                    #     print('fail',idx, iteration,i_loss,p_loss)
                if iteration == (self.iters-1) and acc.item() < 99:
                    print('error',idx, iteration, i_loss,p_loss )
                    # print(acc)
                

                steps += 1
                # update progress bar
                bar.update()
                # clear cuda cache
                torch.cuda.empty_cache()
                #print(f"Inverted stats: mean: {inputs.mean()} -- std: {inputs.std()}")
                self.inverted_pts.append(inputs.data.cpu().numpy()[0])
                self.inverted_targets.append(targets.cpu().numpy()[0])

            bar.close() 

            # print("Check accuracy with the inverted samples:")

            # # Inverted input accuracy
            # num_samples_batch = inversion_batch_size
            # for batch_id in range(num_batches):
            #     if batch_id == num_batches - 1:
            #         num_samples_batch = size - (batch_id * num_samples_batch)
            #     inputs_list = self.inverted_pts[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            #     targets_list = self.inverted_targets[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            #     inputs = torch.zeros(size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
            #     targets = torch.zeros(size=(num_samples_batch,)).cuda()
            #     for i in range(len(inputs_list)):
            #         inputs[i] = torch.from_numpy(inputs_list[i])
            #         targets[i] = targets_list[i]  
            #     with torch.inference_mode():
            #         outputs = self.solver(inputs)
            #         acc = acc_meter(outputs, targets) * 100
            #         self.inverted_outputs.extend(outputs.cpu().numpy())
            # print(f"Acc inverted samples = {acc_meter.compute() * 100}")
            # acc_meter.reset()

            #save inverted samples
            # fname = "saved_inverted_sample_prior" + '.pkl'
            fname = "saved_inverted_sample_" +str(self.step-1)+ '.pkl'
            fpath = osp.join(inverted_sample_dir, fname)
            if self.num_known_classes !=0:
                print("load saved data")
                saved_data  = utils.load_pickle(fpath)
                self.inverted_pts.extend(saved_data["pts"])
                self.inverted_targets.extend(saved_data["target"])
                self.inverted_outputs.extend(saved_data["output"])
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
             y_hat, features, feature_cl = self.solver.forward_feature(x)

        return features
    def generate_scores_pen_ib(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            features,z = self.solver.forward_feature(x)

        return z

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)   

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)
def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")
def diff_entropy2(x, k=3, base=2):
    """The classic K-L k-nearest neighbor continuous entropy estimator
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def diff_entropy(var):
            n = var.shape[0]
            # print(n)
            det = np.linalg.det(var)
            entropy = 0.5 * ((np.log(2 * np.pi) + 1) * n + np.log(det))
            return entropy

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

def get_prototype(data,class_size,pca_param) :
        proto_mean, proto_var, feat_dict = get_mean_cov(data)
        #eig_decomposition 
        reduced_var = {}
        reduced_mean = {}
        reduced_feat_dict ={}
        directions_dict =  {}
        u_dict = {}
        for c in proto_mean:
            eig_vecs, _, n_keep, eig_va = utils.get_eig_vecs(proto_var[c], pca_param); #0.95
            # n_keep = 3
            U = eig_vecs[:, :n_keep]
            # print(proto_mean[c].shape)
            reduced_feat_dict[c] = (feat_dict[c]-proto_mean[c]).dot(U)
            # print(reduced_feat_dict[c].shape)
            reduced_var[c] = np.cov(reduced_feat_dict[c].T)
            reduced_mean[c] = np.mean(reduced_feat_dict[c].T,axis = 1)
            d, n = U.shape;
        
            directions = np.diag( [1]*n)
            print(reduced_var[c].shape)
            print(n)
            principal_axes = utils.get_mixture_of_samples(directions, n, n, class_size, 3);
            directions_dict[c] = principal_axes
            u_dict[c] = U
            
            # print(reduced_mean[c].shape)    
        
        return proto_mean, proto_var,reduced_mean,reduced_var,feat_dict, reduced_feat_dict, directions_dict,u_dict

def get_norm_dist(pred, target) :
        norm_ = torch.norm(target).item();
        # print(torch.dist(pred, target))
        d = torch.dist(pred, target) / norm_;
        return d;
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2])
        var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()



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
    