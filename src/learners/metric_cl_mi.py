# #linear 
import sys
import importlib
import shutil
from tqdm import tqdm
import torch
from torch.nn import functional as F
from easydict import EasyDict as edict
import torchmetrics
from torch.optim import Adam
import numpy as np  
import os, os.path as osp
import copy

import optimizers as optimizer_defs
import utils
import model_defs
from .base import Base
from .helpers import *
from .helpers2 import *
import losses as loss_defs

import logging 

# #now we will Create and configure logger 
# logging.basicConfig(filename="/home/luzhenyu/DFCIL/guided_MI/output/hgr_shrec_2017/Wts/trial_1/out.log", 
# 					format='%(asctime)s %(message)s', 
# 					filemode='w') 

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

class Metric_CL_MI(Base):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(Metric_CL_MI, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
        self.inversion_replay = False
        self.KD_replay = False
        self.power_iters = self.cfg.increm.learner.power_iters
        self.deep_inv_params = self.cfg.increm.learner.deep_inv_params
        self.gen_inverted_samples = False

        # gen parameters
        self.generator = None
        self.generator_optimizer = None
        # self.generator = self.create_generator()
        # self.generator_optimizer = Adam(params=self.generator.parameters(), lr=self.deep_inv_params[0])

    def train(self, n_trial):
        print(f"Using GPU = {self.args.gpu} with (batch_size, workers) = ({self.cfg.batch_size}, {self.cfg.workers})")
        torch.cuda.set_device(self.args.gpu)

        self.cfg.num_total_classes = self.cfg_data.get_n_classes(self.args.split_type)
        # Load model
        self.model,self.classifier = model_defs.get_model(edict({'n_classes': self.cfg.num_total_classes,**self.cfg.model}))
        
        # Class mapping vars
        c = 0
        self.cfg.class_mapping = {}
        label_to_name = self.cfg_data.label_to_name[self.args.split_type]
        self.cfg.label_to_name_mapped = {}
        # Run tasks
        for current_t_index in range(self.cfg.increm.max_task):
            # print name
            train_name = str(current_t_index)
            print('======================', train_name, '=======================')
            # Set variables depending on the task
            if current_t_index > 0:
                total_epochs_task = self.cfg.total_epochs_incremental_task
                self.cfg.total_epochs = self.cfg.total_epochs_incremental_task
                self.known_classes = self.valid_out_dim
                self.add_classes = self.cfg.increm.other_split_size
                self.valid_out_dim += self.cfg.increm.other_split_size
            else:
                total_epochs_task = self.cfg.total_epochs
                self.valid_out_dim = self.cfg.increm.first_split_size
                self.known_classes = 0
                self.add_classes = self.valid_out_dim
                

            # Load best checkpoint if desired. Otherwise, continue training from last checkpoint
            if current_t_index == 1 and self.cfg.increm.load_best_checkpoint_train:
                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_{current_t_index - 1}"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                del checkpoint

            model_defs.print_n_params(self.model)
            best_measure_info = utils.init_best_measure_info('acc', 'accuracy')
            log_dir_task = osp.join(self.args.log_dir, f"task_{train_name}")

            # load dataset for task
            # self.train_dataset = getattr(importlib.import_module('.' + self.args.dataset +'_multiview', package='datasets'), 
            #         'Dataset')('train', self.args.split_type, self.cfg.n_views, self.cfg_data, self.cfg.transforms['train'], 
            #         self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial,few_shot_size = self.cfg.few_shot_size if current_t_index > 0 else -1)

            # self.train_dataset_no_cts = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
            #         'Dataset')('train', self.args.split_type, self.cfg_data, self.cfg.transforms['train'], 
            #         self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial,few_shot_size = self.cfg.few_shot_size if current_t_index > 0 else -1)      
                    
            self.train_dataset = getattr(importlib.import_module('.' + self.args.dataset +'_multiview', package='datasets'), 
                    'Dataset')('train', self.args.split_type, self.cfg.n_views, self.cfg_data, self.cfg.transforms['train'], 
                    self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

            self.train_dataset_no_cts = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
                    'Dataset')('train', self.args.split_type, self.cfg_data, self.cfg.transforms['train'], 
                    self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)      
                    

            self.val_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
                    'Dataset')('val', self.args.split_type, self.cfg_data, self.cfg.transforms['val'], 
                               
                     self.valid_out_dim, 0, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)
            print(f"Training classes: {self.train_dataset.keep_class_l}")
            # Class and label mapping
            for k in self.train_dataset.keep_class_l:
                self.cfg.class_mapping[str(k)] = c
                c += 1
            for prev_class, new_class in self.cfg.class_mapping.items():
                self.cfg.label_to_name_mapped[str(new_class)] = label_to_name[int(prev_class)]
            
            self.train_sampler = None
            self.val_sampler = None
            # Append coreset samples to the train/val datasets if memory > 0
            self.train_dataset.append_coreset(self.coreset_train, self.ic, only=False)

            g = torch.Generator()
            g.manual_seed(3407)

            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=(self.train_sampler is None),
                num_workers=self.cfg.workers, pin_memory=True, sampler=self.train_sampler, drop_last=True if self.n_gpus>1 else False,
                worker_init_fn=utils.seed_worker, generator=g)
            
            self.train_loader_no_cts = DataLoader(self.train_dataset_no_cts, batch_size=self.cfg.batch_size, shuffle=(self.train_sampler is None),
                num_workers=self.cfg.workers, pin_memory=True, sampler=self.train_sampler, drop_last=True if self.n_gpus>1 else False,
                worker_init_fn=utils.seed_worker, generator=g)

            
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=(self.val_sampler is None),
                num_workers=self.cfg.workers, pin_memory=True, sampler=self.val_sampler, drop_last=True if self.n_gpus>1 else False,
                worker_init_fn=utils.seed_worker, generator=g)   
            

            
            if current_t_index == 0 and self.cfg.increm.load_pretrained_task0:
                # Load pretrained model for task 0
                print("Loading pretrained model for task 0")
                self.valid_out_dim = self.cfg.increm.first_split_size
                # Create log dir if it does not exist
                if not osp.exists(osp.join(log_dir_task, 'checkpoints')):
                    os.makedirs(osp.join(log_dir_task, 'checkpoints'))
                # Copy checkpoint to log dir
                    
                #TODO change back to pretrained path
                    
                # pretrained_checkpoint_path = osp.join('/home/luzhenyu/DFCIL/guided_MI/models', self.args.dataset,'dg_sta_wts_cts', f"trial_{n_trial+1}", 'checkpoints', 'model_best.pth.tar')
                pretrained_checkpoint_path ='/home/luzhenyu/DFCIL/guided_MI/output2/hgr_shrec_2017/Metric-CL-infonce/trial_1/task_0/checkpoints/checkpoint_100.pth.tar'
                # pretrained_checkpoint_path ='/home/luzhenyu/DFCIL/guided_MI/output2/ego_gesture/Metric-CL-infonce/trial_3/task_0/checkpoints/checkpoint_100.pth.tar'

                assert osp.exists(pretrained_checkpoint_path), f"Pretrained checkpoint not found in {pretrained_checkpoint_path}"
                shutil.copy(pretrained_checkpoint_path, osp.join(log_dir_task, 'checkpoints', 'model_best.pth.tar'))

                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_0"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                self.model.cuda(self.args.gpu)
                
            else:

                # Save a sample of the dataset
                # utils.display_sample(pts=self.val_dataset[0]['pts'], target=self.val_dataset[0]['label'], label2name=self.cfg_data.label_to_name)
                # Generate inverted samples
                if self.gen_inverted_samples:

                    #TODO determine whether to use old features
                    # self.generate_inverted_samples(self.previous_teacher, self.cfg.increm.learner.n_samples_per_class, self.cfg.increm.learner.inversion_batch_size, self.cfg.batch_size*8, log_dir_task, self.args.log_dir)
                    
                    
                    self.generate_inverted_samples(self.previous_teacher, self.cfg.increm.learner.n_samples_per_class, self.cfg.increm.learner.inversion_batch_size, self.cfg.batch_size*8, log_dir_task, self.args.log_dir)

                # Modify the base LR if current_task_index > 0
                if current_t_index > 0:
                    self.cfg.optimizer.lr = self.cfg.optimizer.lr_incremental_task
                    if not self.cfg.optimizer.include_scheduler and 'scheduler' in self.cfg.optimizer :
                        del self.cfg.optimizer.scheduler
                        del self.cfg.optimizer['scheduler']
                        self.cfg.step_per_epoch = False
                        self.cfg.step_per_batch = False
                        print("Scheduler deleted")
                    
                for param in self.model.parameters():
                    param.requires_grad = True  
                    # for name, param in self.model.named_parameters():
                    #     if 'pre_feat' in name or 'cl_head' in name:  # Check if the parameter belongs to the desired part of the layer
                     
                    #         param.requires_grad = True     # Set requires_grad to True for this parameter
                    #     else:
                    #         param.requires_grad = False    # Set requires_grad

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter '{name}' requires gradients.")
                #     else:
                #         print(f"Parameter '{name}' does not require gradients.")
                self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(self.model, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )

                self.criteria = loss_defs.get_losses(self.cfg.loss, self.valid_out_dim)
                if 'rkd' in self.criteria:
                    module = nn.ModuleList([self.model, self.criteria['rkd'].func])
                    self.criteria['rkd'].func.cuda(self.args.gpu)
                    self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(module, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )
                else:
                    self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(self.model, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )
                resume_checkpoint_path = utils.get_last_checkpoint_path(log_dir_task)
                if resume_checkpoint_path :
                    print(f"=> loading checkpoint {resume_checkpoint_path}")
                    checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'))
                    start_epoch = checkpoint['epoch'] + 1
                    if start_epoch >= total_epochs_task :
                        print(f"Start epoch {start_epoch} is greater than total epochs {total_epochs_task}")
                        sys.exit()
                    utils.load_state_dict_single(checkpoint['state_dict'], self.model, self.optimizer, self.scheduler, )
                    print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                    del checkpoint

                else :
                    start_epoch = 1
                    print("=> no checkpoint found for resuming.")

                # Freeze backbone if desired (from the second task onwards)
                if current_t_index > 0 and self.cfg.increm.freeze_feature_extractor:
                    print("Freezing feature extractor...")
                    self.freeze_model(feature_extractor=True)

                # Freeze the weights for the previous classes in the classification layer if desired (from the second task onwards)
                if current_t_index > 0 and self.cfg.increm.freeze_classifier:
                    # Copy the weights and biases of the final linear layer
                    self.prev_weights = torch.empty_like(self.model.final.weight.data).copy_(self.model.final.weight.data)
                    self.prev_bias = torch.empty_like(self.model.final.bias.data).copy_(self.model.final.bias.data)

                # transfer models
                self.model.cuda(self.args.gpu)
                
                # transfer optimizers and schedulers
                optimizer_defs.optimizer_to_cuda(self.optimizer, self.args.gpu)
                optimizer_defs.scheduler_to_cuda(self.scheduler, self.args.gpu)

             
                train_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'train'))
                val_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'val'))

                # epoch, train, val bars
                print('Printing progress info for GPU 0 only ...')
                ebar = tqdm(total=total_epochs_task - start_epoch + 1, leave=True, desc='epoch', dynamic_ncols=False)
                tbar = tqdm(total=len(self.train_loader), leave=True, desc='train', dynamic_ncols=False)
                vbar = tqdm(total=len(self.val_loader), leave=True, desc='val', dynamic_ncols=False)

                step_per_epoch = False
                if 'scheduler' in self.cfg.optimizer :
                    if 'step_per_epoch' in self.cfg.optimizer.scheduler :
                        step_per_epoch = self.cfg.optimizer.scheduler.step_per_epoch

                for epoch in range(start_epoch, total_epochs_task + 1) :
                    torch.cuda.empty_cache()

                    self.train_epoch(tbar , epoch, train_logger , current_t_index)

                    # measures = self.validate_epoch(vbar if self.args.gpu==0 else None, epoch, val_logger if self.args.gpu==0 else None) 

        
                        # is_best = best_measure_info.func(measures[best_measure_info.tag],best_measure_info.val)
                        # if is_best :
                        #     best_measure_info.val = measures[best_measure_info.tag]            

                    train_logger.flush()
                    val_logger.flush()           

                    if (epoch % self.args.save_epoch_freq == 0):
                        # save model 
                        state_dict = utils.get_state_dict_single(self.model, self.optimizer, self.scheduler, self.is_distributed)

                        utils.save_checkpoint(log_dir_task,
                            { 
                                'epoch': epoch, 
                                'state_dict': state_dict, 
                                'best_measure_tag': best_measure_info.tag,
                                'best_measure': best_measure_info.val, 
                            },
                            epoch,
                            save_last_only=self.args.save_last_only,
                            is_best=False,
                        )
                        # monitor feature space change
                    self.get_feature_w_syn(0,current_t_index,1,epoch)    

                    if step_per_epoch :
                        optimizer_defs.step_scheduler(self.scheduler)

           
                    ebar.update()
                    ebar.set_postfix(dict(epoch=epoch)) 
                
            
            #TODO train classifer
            classifer_type = 'linear' #'svm'
            if classifer_type !='svm':
                # if current_t_index == 0:
                _ , self.classifier = model_defs.get_model(edict({'n_classes': self.cfg.num_total_classes,**self.cfg.model}))
                self.classifier.cuda(self.args.gpu)
                self.optimizer2, self.scheduler2 = optimizer_defs.get_optimizer_scheduler(self.classifier, edict({**self.cfg.optimizer, 
                                            'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}), True )
                self.criteria = loss_defs.get_losses(self.cfg.loss, self.valid_out_dim)
        
                train_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'train'))
                val_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'val'))

                # epoch, train, val bars
                print('Printing progress info for GPU 0 only ...')
                ebar = tqdm(total=20 + 1, leave=True, desc='epoch', dynamic_ncols=False)
                tbar = tqdm(total=len(self.train_loader), leave=True, desc='train', dynamic_ncols=False)
                vbar = tqdm(total=len(self.val_loader), leave=True, desc='val', dynamic_ncols=False)
                if current_t_index == 0:
                    t = 50
                else:
                    t =50
                best = 0
                for epoch in range(0, t) :
                    # for epoch in range(0, total_epochs_task + 1) :
                    torch.cuda.empty_cache()
                    for param in self.model.parameters():
                        param.requires_grad = False
                    self.train_epoch_classify(tbar , epoch, train_logger, current_t_index)
                    result = self.validate_epoch_classify(tbar , epoch, val_logger )
                    if result['acc'] > best:
                        best = result['acc']
                        best_per_class = result ['per_class_acc']
                print(best)
                print(best_per_class)
                    # if step_per_epoch :
                    #     optimizer_defs.step_scheduler(self.scheduler)

                    # if self.args.gpu == 0 :
                    #     ebar.update()
                    #     ebar.set_postfix(dict(epoch=epoch))  
                    #     if (epoch % self.args.save_epoch_freq == 0) and (self.args.gpu == 0) :
                    #         # save model 
                    #         state_dict = utils.get_state_dict_single(self.classifier, self.optimizer2, self.scheduler2, self.is_distributed)

                    #         utils.save_checkpoint(log_dir_task,
                    #             { 
                    #                 'epoch': epoch, 
                    #                 'state_dict': state_dict, 
                    #                 'best_measure_tag': best_measure_info.tag,
                    #                 'best_measure': best_measure_info.val, 
                    #             },
                    #             epoch,
                    #             save_last_only=self.args.save_last_only,
                    #             is_best=False,
                    #         )
            else:
                self.train_classify_svm()
                self.validate_classify_svm()

            self.KD_replay = True
            self.last_valid_out_dim = self.valid_out_dim
            
            # set to eval mode
            self.model.eval()
            # new teacher

            # self.get_feature(0)
            # self.get_val_feature(0,current_t_index)
            self.get_feature_w_syn(0,current_t_index, 1, -1)
            self.sample_shape = (-1, self.cfg.seq_len, self.cfg.model.n_joints, self.cfg.in_channels)
            self.previous_teacher = Teacher_v4(solver=copy.deepcopy(self.model), 
                                                classifier=copy.deepcopy(self.classifier),
                                                sample_shape = self.sample_shape, 
                                                iters = self.power_iters, 
                                                deep_inv_params = self.deep_inv_params, 
                                                class_idx = np.arange(self.valid_out_dim), 
                                                num_inverted_class = self.add_classes,
                                                num_known_classes = self.known_classes,
                                                config = self.cfg, step = current_t_index)
            #uncomment
            
            self.previous_teacher.generate_target_feature(self.args.log_dir)
            self.inversion_replay = True
            self.gen_inverted_samples = True
        

            # Update coreset for train/val datasets if memory > 0
            if self.cfg.increm.memory > 0:
                self.coreset_train = self.train_dataset.update_coreset(self.coreset_train, self.cfg.increm.memory, 
                    np.arange(self.last_valid_out_dim), self.cfg.class_mapping)


        if not (current_t_index == 0 and self.cfg.increm.load_pretrained_task0):
            ebar.close()
            tbar.close()
            vbar.close()
            train_logger.close()
            val_logger.close()
    

        # save config 

            # Save config edict object 
        utils.stdio.save_pickle(osp.join(self.args.log_dir, 'config.pkl'), self.cfg)

 
    
    def train_epoch(self, tbar, epoch, train_logger, current_t_index) :
        
        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics 
        acc_meter_act = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_gen = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_global = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim, average=None).cuda(self.args.gpu)

        n_batches = len(self.train_loader)

        # set to train mode
        self.model.train()

        # set epochs
        if self.train_sampler is not None :
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)


        tbar.reset(total=n_batches)
        tbar.refresh()

        step_per_batch = False
        if 'scheduler' in self.cfg.optimizer :
            if 'step_per_batch' in self.cfg.optimizer.scheduler :
                step_per_batch = self.cfg.optimizer.scheduler.step_per_batch

        iter_loader = iter(self.train_loader)
        bi = 1
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            bs, n_views = pts.shape[:2];
            # pts = pts.view(bs * n_views, *pts.shape[2:]);
            pts =  torch.cat(torch.unbind(pts, dim=1), dim=0)
            # print(pts.shape)
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            target = torch.cat([target]* n_views)
            # if synthetic data replay
            if self.inversion_replay:
                pts_replay, target_replay, target_replay_hat = self.sample(self.previous_teacher)
                # Send to GPU
                pts_replay = pts_replay.cuda(self.args.gpu)
                target_replay = target_replay.cuda(self.args.gpu)
                target_replay_hat = target_replay_hat.cuda(self.args.gpu)

            # self.KD_replay = False
            # # if KD -> generate pseudo labels
            # if self.KD_replay:
            #     allowed_predictions = np.arange(self.last_valid_out_dim)
            #     feat_replay_hat = self.previous_teacher.generate_scores_pen(pts_replay)
            #     # _, target_hat_com = self.combine_data(((pts, target_hat),(pts_replay, target_replay_hat[:,:self.last_valid_out_dim])))
            # else:
            #     target_hat_com = None
            feat_replay_hat = None
            # combine inputs and generated samples for classification
            if self.inversion_replay:
                pts_com, target_com = self.combine_data(((pts, target),(pts_replay, target_replay)))
                # pts_com, target_com = pts_replay, target_replay
                feat_replay_hat, feat_replay_hat_cl = self.previous_teacher.generate_scores_pen(pts_com)
            else:
                pts_com, target_com = pts, target
            
            feature, feature_cl = self.model.forward_feature(pts_com)


            
            # feature = feature.view(bs, n_views, *feature.shape[1:]);
            # feature_cl = feature_cl.view(bs, n_views, *feature_cl.shape[1:]);
            # output = output[:, :self.valid_out_dim]
            #batch_idx = np.arange(self.cfg.batch_size)
            # print(target_com)
            new_idx = np.arange(len(pts))
            #used to kd
            old_idx = np.arange(len(pts), len(pts_com))
            # print (target_com[old_idx])
            # print(kd_index)
            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'rkd' :
                    if feat_replay_hat == None:
                        continue
                    # print(features[kd_index].shape)
                    # print(feat_replay_hat[kd_index].shape)
                 
                    lval = lfunc(feature[old_idx], feat_replay_hat[old_idx])
                    losses[lname].update(lval.item(), feature[old_idx].size(0) )

                    loss_tensors.append(lweight * lval)
                elif lname == 'cross_entropy':
                    continue
                else:
                    # old_feature = feature_cl[old_idx]
                    # old_target = target[old_idx]
                    # new_feature = feature_cl[new_idx]
                    # new_target = target[new_idx]
                    lval = lfunc(feature_cl,target_com,new_idx,old_idx)
                    # lval = lfunc(feature_cl,target_com)#,new_idx,old_idx)
                    # print(lval)
                    losses[lname].update(lval.item(), feature_cl.size(0) )
                    loss_tensors.append(lweight * lval)
                # else:
                #     lval = lfunc(output, target_com)
                #     # lval = lfunc(output[batch_idx], target_com[batch_idx])
                #     losses[lname].update(lval, output.size(0) )
                #     loss_tensors.append(lweight * lval)
                    # print(lval)

            loss = sum(loss_tensors)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01);
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.00005);
            
            self.optimizer.step()

            if  self.cfg.increm.freeze_classifier and current_t_index > 0:
                # Restore the weights and biases for previous classes
                self.model.final.weight.data[:self.last_valid_out_dim] = self.prev_weights[:self.last_valid_out_dim]
                self.model.final.bias.data[:self.last_valid_out_dim] = self.prev_bias[:self.last_valid_out_dim]

            # if self.inversion_replay:
            #     train_acc_actual = acc_meter_act(output[:len(target)], target_com[:len(target)]) * 100
            #     if len(kd_index) > 0:
            #         train_acc_generated = acc_meter_gen(output[len(target):], target_com[len(target):]) * 100
            # train_acc_global = acc_meter_global(output, target_com) * 100
            # print(train_acc_global)
            if step_per_batch :
                optimizer_defs.step_scheduler(self.scheduler)


            # if self.args.gpu == 0 :           
            #     tbar.update()
            #     tbar.set_postfix({
            #             'it': bi,
            #             'loss': loss.item(), 
            #             'train_acc_global':torch.mean(train_acc_global),
            #     })
            #     tbar.refresh()

            bi += 1
        # self.get_feature(tbar)

            # hyperparam update
            # train_logger.update(
            #     {'learning_rate': self.optimizer.param_groups[0]['lr']},
            #     step=epoch, prefix="stepwise");

            # loss update
        train_logger.update(
            { ltype: lmeter.avg for ltype, lmeter in losses.items() },
            step=epoch, prefix="loss");
        return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }   
        print(return_values) 
        train_logger.flush();              

        tbar.update();
        tbar.set_postfix({
                'it': bi,
                'loss': loss.item(), 
        });
        tbar.refresh();
        # if self.args.gpu == 0 :
        #     acc_all = acc_meter_global.compute() * 100

        #     # hyperparam update
        #     train_logger.update(
        #         {'learning_rate': self.optimizer.param_groups[0]['lr']},
        #         step=epoch, prefix="stepwise")

        #     # loss update
        #     train_logger.update(
        #         { ltype: lmeter.avg for ltype, lmeter in losses.items() },
        #         step=epoch, prefix="loss")

        #     # measures update
        #     train_logger.update({
        #         'global': torch.mean(acc_all),
        #         }, step=epoch, prefix="acc" )  

        #     if self.inversion_replay:
        #         acc_actual = acc_meter_act.compute() * 100
        #         train_logger.update({
        #             'actual': acc_actual,
        #             }, step=epoch, prefix="acc" )
        #         acc_meter_act.reset()
        #         acc_generated = acc_meter_gen.compute() * 100
        #         train_logger.update({
        #             'generated': acc_generated,
        #             }, step=epoch, prefix="acc" )  
        #         acc_meter_gen.reset()

        #     acc_meter_global.reset()

        #     train_logger.flush()  

              
    @torch.no_grad()
    def validate_epoch(self, vbar, epoch, val_logger) :
   
        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim,average = None).cuda(self.args.gpu)
        # set to eval mode
        self.model.eval()


        vbar.reset(total=len(self.val_loader))
        vbar.refresh()

        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            bs, n_views = pts.shape[:2];
            pts = pts.view(bs * n_views, *pts.shape[2:]);
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = list(range(self.last_valid_out_dim))
                # output_hat = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
            else:
                output_hat = None

        
            output,feature = self.model.forward_feature(pts_com)
            # output = output[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'rkd' :
                    continue
                    loss_tensors.append(lweight * lval)
                # else:
                #     lval = lfunc(output,target)
                #     losses[lname].update(lval.item(), output.size(0) )
                #     loss_tensors.append(lweight * lval)
            loss = sum(loss_tensors)
            
        #     val_acc= acc_meter(output, target) * 100
        #     acc_meter_per_class(output, target)
        #     # print('val_acc_batch',val_acc)
        #     # val_acc = torch.mean(val_acc)
        #     if self.args.gpu == 0 :
        #         vbar.update()
        #         vbar.set_postfix({
        #             'it': bi,
        #             'loss': loss.item(), 
        #             'val_acc': val_acc.item(),
        #         })                
        #         vbar.refresh()

        #     bi += 1   

        # if self.args.gpu == 0 :
        #     acc_all = acc_meter.compute() * 100
        #     per_class_accuracy = acc_meter_per_class.compute()
        #     print('per_class_accuracy',per_class_accuracy)
        #     print(acc_all)
        #     # loss update
        #     val_logger.update(
        #         { ltype: lmeter.avg for ltype, lmeter in losses.items() },
        #         step=epoch, prefix="loss")

        #     # measures update
        #     val_logger.update({
        #         'global': acc_all,
        #         }, step=epoch, prefix="acc" ) 
                                 
        #     acc_meter.reset()   
        #     val_logger.flush() 

        #     return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
        #     return_values['acc'] = acc_all
            
        #     return return_values


    def train_epoch_classify(self, tbar, epoch, train_logger, current_t_index) :

        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics 
        acc_meter_act = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_gen = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_global = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim, average=None).cuda(self.args.gpu)

        n_batches = len(self.train_loader)

        # set to train mode
        # self.model.zero_grad()
        self.model.eval()
        # for name, param in self.model.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")
        
        self.classifier.train()

        # set epochs
        if self.train_sampler is not None :
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)

        tbar.reset(total=n_batches)
        tbar.refresh()

        step_per_batch = False
        if 'scheduler' in self.cfg.optimizer :
            if 'step_per_batch' in self.cfg.optimizer.scheduler :
                step_per_batch = self.cfg.optimizer.scheduler.step_per_batch

        iter_loader = iter(self.train_loader)
        bi = 1
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            bs, n_views = pts.shape[:2];
            pts =  torch.cat(torch.unbind(pts, dim=1), dim=0)
            # # print(pts.shape)
            # # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]
            target = torch.cat([target]* n_views)

            if self.inversion_replay:
                pts_replay, target_replay, target_replay_hat = self.sample(self.previous_teacher)
                # Send to GPU
                pts_replay = pts_replay.cuda(self.args.gpu)
                target_replay = target_replay.cuda(self.args.gpu)
                target_replay_hat = target_replay_hat.cuda(self.args.gpu)
               


            feat_replay_hat = None
            # combine inputs and generated samples for classification
            if self.inversion_replay:
                pts_com, target_com = self.combine_data(((pts, target),(pts_replay, target_replay)))
                # pts_com, target_com = pts_replay, target_replay
                feat_replay_hat, feat_replay_hat_cl = self.previous_teacher.generate_scores_pen(pts_com)
            else:
                pts_com, target_com = pts, target

            # print(target_com)
            with torch.no_grad():
                feature, feature_cl = self.model.forward_feature(pts_com)
            # output = feature.view(bs, n_views, *feature.shape[1:]);
            # # output = output[:, 0, :]
            # output = torch.cat(torch.unbind(output, dim=1), dim=0)
            output = self.classifier(feature)

            output = output[:, :self.valid_out_dim]
            # print(output.size())
            #batch_idx = np.arange(self.cfg.batch_size)
            # batch_idx = np.arange(len(pts))
            # kd_index = np.arange(len(pts), len(pts_com))
            # print(kd_index)
            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'cross_entropy':
                    # print('mmd')
                    # print(self.model.z_prior.shape)
                    lval = lfunc(output,target_com)
                    losses[lname].update(lval.item(), output.size(0) )
                    loss_tensors.append(lweight * lval)
                # else:
                #     lval = lfunc(output, target_com)
                #     # lval = lfunc(output[batch_idx], target_com[batch_idx])
                #     losses[lname].update(lval, output.size(0) )
                #     loss_tensors.append(lweight * lval)
                    # print(lval)


            pre_update_values = {name: param.clone() for name, param in self.model.named_parameters()}


            loss = sum(loss_tensors)

            self.optimizer2.zero_grad()
            loss.backward()

            self.optimizer2.step()


            # for name, param in self.model.named_parameters():
            #     pre_value = pre_update_values[name]
            #     updated = not torch.equal(pre_value, param)
            #     print(f"{name} updated: {updated}")

            train_acc_global = acc_meter_global(output, target_com) * 100
            # print('training_acc', train_acc_global)
            if step_per_batch :
                optimizer_defs.step_scheduler(self.scheduler2)


        
            tbar.update()
            tbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(), 
                    'train_acc_global':torch.mean(train_acc_global),
            })
            tbar.refresh()

            bi += 1
        # if self.args.gpu == 0 :
        #     acc_all = acc_meter_global.compute() * 100

        #     # hyperparam update
        #     train_logger.update(
        #         {'learning_rate': self.optimizer.param_groups[0]['lr']},
        #         step=epoch, prefix="stepwise")

        #     # loss update
        #     train_logger.update(
        #         { ltype: lmeter.avg for ltype, lmeter in losses.items() },
        #         step=epoch, prefix="loss")

        #     # measures update
        #     train_logger.update({
        #         'global': torch.mean(acc_all),
        #         }, step=epoch, prefix="acc" )  

        #     if self.inversion_replay:
        #         acc_actual = acc_meter_act.compute() * 100
        #         train_logger.update({
        #             'actual': acc_actual,
        #             }, step=epoch, prefix="acc" )
        #         acc_meter_act.reset()
        #         acc_generated = acc_meter_gen.compute() * 100
        #         train_logger.update({
        #             'generated': acc_generated,
        #             }, step=epoch, prefix="acc" )  
        #         acc_meter_gen.reset()

        #     acc_meter_global.reset()

        #     train_logger.flush() 
            
 

    @torch.no_grad()
    def validate_epoch_classify(self, vbar, epoch, val_logger) :
   
        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        acc_meter_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim,average = None).cuda(self.args.gpu)
        # set to eval mode
        self.model.eval()
        self.classifier.eval()

        vbar.reset(total=len(self.val_loader))
        vbar.refresh()

        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = list(range(self.last_valid_out_dim))
                # output_hat = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
            else:
                output_hat = None

        
            feature, feature_cl = self.model.forward_feature(pts)
            output = self.classifier(feature)
            output = output[:, :self.valid_out_dim]
            # print(target)
            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'mse' :
                    continue
                    loss_tensors.append(lweight * lval)
                elif lname =='cross_entropy':
                    lval = lfunc(output,target)
                    losses[lname].update(lval.item(), output.size(0) )
                    loss_tensors.append(lweight * lval)
            loss = sum(loss_tensors)
            
            val_acc= acc_meter(output, target) * 100
            acc_meter_per_class(output, target)
            # print('val_acc_batch',val_acc)
        #     # val_acc = torch.mean(val_acc)
  
            vbar.update()
            vbar.set_postfix({
                'it': bi,
                'loss': loss.item(), 
                'val_acc': val_acc.item(),
            })                
            vbar.refresh()

            bi += 1   


        acc_all = acc_meter.compute() * 100
        per_class_accuracy = acc_meter_per_class.compute()
        # print('per_class_accuracy',per_class_accuracy)
        # print(acc_all)
        # loss update
        val_logger.update(
            { ltype: lmeter.avg for ltype, lmeter in losses.items() },
            step=epoch, prefix="loss")

        # measures update
        val_logger.update({
            'global': acc_all,
            }, step=epoch, prefix="acc" ) 
                                
        acc_meter.reset()   
        val_logger.flush() 

        return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
        return_values['acc'] = acc_all
        return_values['per_class_acc'] = per_class_accuracy
        return return_values

    
    # def train_classify_svm(self) :


    #     n_batches = len(self.train_loader)

    #     # set to train mode
    #     self.model.eval()
    #     self.classifier.train()

    #     # set epochs
    #     if self.train_sampler is not None :
    #         self.train_sampler.set_epoch(self.train_sampler.epoch + 1)


    #     iter_loader = iter(self.train_loader)
    #     bi = 1

    #     X_train, y_train = [], [];
    #     while bi <= n_batches :
    #         data = next(iter_loader)
    #         # transfer data to gpu
    #         utils.tensor_dict_to_cuda(data, self.args.gpu)

    #         pts, target = data.pts, data.label
    #         bs, n_views = pts.shape[:2];
    #         pts = pts.view(bs * n_views, *pts.shape[2:]);
    #         # # print(pts.shape)
    #         # # Map target
    #         for i, target_class in enumerate(target):
    #             target[i] = self.cfg.class_mapping[str(target_class.item())]
    #         target = torch.cat([target,target,target]) 

    #         with torch.no_grad():
    #             feature, feature_cl = self.model.forward_feature(pts)
    #         output = feature.view(bs, n_views, *feature.shape[1:]);
    #         # output = output[:, 0, :]
    #         output = torch.cat(torch.unbind(output, dim=1), dim=0)
    #         output = output.data.cpu().numpy();
    #         label = target.cpu().numpy();
    #         X_train.append(output);
    #         y_train.append(label);
          
    #         bi += 1
        
    #     X_train = np.concatenate(X_train, axis=0);
    #     y_train = np.concatenate(y_train, axis=0);
    #     self.classifier = utils.get_svm_classifier('rbf');
    #     print("Fitting classifier to data ...")
    #     print(f"Shape of data = {X_train.shape}"); 
    #     utils.fit_classifier(self.classifier, X_train, y_train);

    #     clf_fpath = osp.join(self.args.log_dir,'classifier.pkl');
    #     print(f"Saving classifier in = {clf_fpath}");
    #     utils.save_pickle(clf_fpath, self.classifier);
    
    def train_classify_svm(self) :

    
        n_batches = len(self.train_loader_no_cts)

        # set to train mode
        self.model.eval()
        self.classifier.train()

        # set epochs
        if self.train_sampler is not None :
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)


        iter_loader = iter(self.train_loader_no_cts)
        bi = 1

        X_train, y_train = [], [];
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # # print(pts.shape)
            # # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]
            # target = torch.cat([target,target,target]) 

            with torch.no_grad():
                feature, feature_cl = self.model.forward_feature(pts)
            # output = output[:, 0, :]
            output = feature_cl.data.cpu().numpy();
            label = target.cpu().numpy();
            X_train.append(output);
            y_train.append(label);
          
            bi += 1
        
        X_train = np.concatenate(X_train, axis=0);
        y_train = np.concatenate(y_train, axis=0);
        self.classifier = utils.get_svm_classifier('rbf');
        print("Fitting classifier to data ...")
        print(f"Shape of data = {X_train.shape}"); 
        utils.fit_classifier(self.classifier, X_train, y_train);

        clf_fpath = osp.join(self.args.log_dir,'classifier.pkl');
        print(f"Saving classifier in = {clf_fpath}");
        utils.save_pickle(clf_fpath, self.classifier);
    


    @torch.no_grad()
    def validate_classify_svm(self) :
        self.model.eval()
  
        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]
    
            features, feature_cl = self.model.forward_feature(pts)
            features = feature_cl.data.cpu().numpy();
            output = utils.predict_classifier(self.classifier, features);
            output = torch.from_numpy(output).long().to(target.device);
            print(output, target)
            bi += 1      
    
        

    def create_generator(self):

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        generator = model_defs.__dict__['generator'].__dict__[self.cfg.increm.learner.gen_model_name](
            self.cfg.seq_len, self.cfg.model.n_joints, self.cfg.in_channels
        )
        return generator        
    

    def sample_generator(self, teacher, dim, return_scores=True):
        return teacher.sample_generator(dim, return_scores=return_scores)

    def sample(self, teacher):
        return teacher.sample()

    def generate_inverted_samples(self, teacher, size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir):
 
        return teacher.generate_inverted_samples_batch(size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir)
        # return teacher.generate_inverted_samples(size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir)
    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y

    @torch.no_grad()
    def get_feature(self, vbar) :
        # save training feature space 
        # set to eval mode
        self.model.eval()
        n_batches = len(self.train_loader)
        iter_loader = iter(self.train_loader)
        bi = 1
        label_list = []
        feat_list = []
        target_list = []
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            bs, n_views = pts.shape[:2];
            pts = pts.view(bs * n_views, *pts.shape[2:]);
            # pts =  torch.cat(torch.unbind(pts, dim=1), dim=0)

            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            target = torch.cat([target]*n_views)
            feature, feature_cl = self.model.forward_feature(pts)

            feature = feature.view(bs, n_views, *feature.shape[1:]);
            # output = output[:, 0, :]
            feature = torch.cat(torch.unbind(feature, dim=1), dim=0)

            output = self.classifier(feature)
            output = output[:, :self.valid_out_dim]
            output = torch.argmax(output,dim = 1)
            feat_list.append(feature.cpu().numpy())
            label_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            bi += 1   

        label_list =np.concatenate(label_list,axis = 0)#.cpu().numpy()
        feat_list = np.concatenate(feat_list, axis = 0)#.cpu().numpy()
        target_list = np.concatenate(target_list, axis = 0)
        save_dict = [label_list,feat_list,target_list]
        utils.stdio.save_pickle(osp.join(self.args.log_dir, 'features.pkl'), save_dict)
    
    @torch.no_grad()
    def get_val_feature(self, vbar, current_t_index) :
        #TODO not complete debug and visulization
        # set to eval mode
        self.model.eval()
        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1
        label_list = []
        feat_list = []
        target_list = []
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            if self.cfg.model.name == 'dg_sta_var':
                output, feature, mu, std = self.model(pts)
            else:
                output,feature= self.model.forward_feature(pts)
            output = output[:, :self.valid_out_dim]
            output = torch.argmax(output,dim = 1)
            feat_list.append(feature.cpu().numpy())
            label_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            bi += 1   

        label_list =np.concatenate(label_list,axis = 0)#.cpu().numpy()
        feat_list = np.concatenate(feat_list, axis = 0)#.cpu().numpy()
        target_list = np.concatenate(target_list, axis = 0)
        save_dict = [label_list,feat_list,target_list]
        utils.stdio.save_pickle(osp.join(self.args.log_dir, 'features_val'+str(current_t_index)+'.pkl'), save_dict)
        

    @torch.no_grad()
    def get_feature_w_syn(self, vbar,current_t_index, viz, epoch) :
        # TODO: used to visualize and debug: generate feature space of training data and syntheic data from previous step
        # set to eval mode
        self.model.eval()
        n_batches = len(self.train_loader)
        iter_loader = iter(self.train_loader)
        bi = 1
        label_list = []
        feat_list = []
        target_list = []
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            bs, n_views = pts.shape[:2];
            # pts = pts.view(bs * n_views, *pts.shape[2:]);
            pts =  torch.cat(torch.unbind(pts, dim=1), dim=0)
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]
            target = torch.cat([target]*n_views)
            if self.inversion_replay:
                pts_replay, target_replay, target_replay_hat = self.sample(self.previous_teacher)
                # print(target_replay)
                # print(pts_replay.shape)
                # Send to GPU
                pts_replay = pts_replay.cuda(self.args.gpu)
                target_replay = target_replay.cuda(self.args.gpu)
                target_replay_hat = target_replay_hat.cuda(self.args.gpu)
            # combine inputs and generated samples for classification
            if self.inversion_replay:
                pts_com, target_com = self.combine_data(((pts, target),(pts_replay, target_replay)))
                feat_replay_hat = self.previous_teacher.generate_scores_pen(pts_com)
                # pts_com, target_com = pts_replay, target_replay
            else:
                pts_com, target_com = pts, target

            feature, feature_cl = self.model.forward_feature(pts_com)
            output = self.classifier(feature)

            output = output[:, :self.valid_out_dim]

            # output = output.view(bs, n_views, *output.shape[1:]);

            # output = output[:, :self.valid_out_dim]
            output = torch.argmax(output,dim = 1)
            feat_list.append(feature.cpu().numpy())
            label_list.append(output.cpu().numpy())
            target_list.append(target_com.cpu().numpy())
            bi += 1   

        label_list =np.concatenate(label_list,axis = 0)#.cpu().numpy()
        feat_list = np.concatenate(feat_list, axis = 0)#.cpu().numpy()
        target_list = np.concatenate(target_list, axis = 0)
        save_dict = [label_list,feat_list,target_list]
        if viz:
            
            viz_dir= osp.join(self.args.log_dir+"/task_"+str(current_t_index), 'features_w_syn'+str(current_t_index) +'_' + str(epoch))
            utils.tsne(feat_list,target_list, viz_dir)
        utils.stdio.save_pickle(osp.join(self.args.log_dir, 'features_w_syn'+str(current_t_index)+'.pkl'), save_dict)
        utils.stdio.save_pickle(osp.join(self.args.log_dir, 'features.pkl'), save_dict)

    
