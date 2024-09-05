import logging




import sys
import os, os.path as osp
import argparse
import yaml
import importlib
from easydict import EasyDict as edict

import torch
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

# Replace sys.stdout with the logging handler


if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils

parser = argparse.ArgumentParser(description='Gesture Recognition.')
parser.add_argument('--train', type=int, default=1, required=False, help='train (1) or testval (0) or test (-1).')
parser.add_argument('--dataset', type=str, default='hgr_shrec_2017', required=False, help='name of the dataset.')
parser.add_argument('--split_type', type=str, default='agnostic', required=False, help='type of data split (if applicable).')
parser.add_argument('--cfg_file', type=str, default='/ogr_cmu/src/configs/params/hgr_shrec_2017/Oracle-BN.yaml', required=False, help='config file to load experimental parameters.')
parser.add_argument('--root_dir', type=str, default='/ogr_cmu/data/SHREC_2017', required=False, help='root directory containing the dataset.')
parser.add_argument('--log_dir', type=str, default='/ogr_cmu/output/hgr_shrec_2017/Oracle-BN', required=False, help='directory for logging.')
parser.add_argument('--save_last_only', action='store_true', help='whether to save the last epoch only.')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='epoch frequency to save checkpoints.')
parser.add_argument('--save_conf_mat', action='store_true', default=1, help='whether to save the confusion matrix.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID.')
parser.add_argument('--trial_id', type=int, default=0, help='trial_ID')
parser.add_argument('--few_shot_seed', type=int, default=1, help='few shot seed id')



#now we will Create and configure logger 
# logging.basicConfig(filename="/home/luzhenyu/DFCIL/guided_MI/output/hgr_shrec_2017/Wts/trial_1/out.log", 
#                     format='%(asctime)s %(message)s', 
#                     filemode='w') 


 
def main() :
    args = parser.parse_args()
    args.dist_url = 'tcp://127.0.0.1:' + utils.get_free_port()
    utils.print_argparser_args(args)
    utils.set_seed()
    n_gpus = torch.cuda.device_count()
    assert n_gpus>0, "A GPU is required for execution."
    main_worker(args.gpu, 1, args)

def main_worker(gpu, n_gpus, args) :
    with open(args.cfg_file, 'rb') as f :
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    cfg_data = getattr(importlib.import_module('.' + args.dataset, package='configs.datasets'),
                'Config_Data')(args.root_dir)   

    is_distributed = False

    is_train = (args.train==1)

    # Learners dict
    learners_dict = {
        'base' : 'Base',
        'base_buffer' : 'Base_Buffer',
        'lwf' : 'LwF',
        'lwf_MC' : 'LwF_MC',
        'deep_inversion' : 'DeepInversion',
        'deep_inversion_gen': 'DeepInversion_gen',
        'abd': 'AlwaysBeDreaming',
        'rdfcil':'Rdfcil',
        'wts':'Wts',
        'wts_ib':'Wts_ib',
        'wts_cts':'Wts_cts',
        'metric_cl':'Metric_CL',
        'metric_ce': 'Metric_CE',
        'metric_cl_mi': 'Metric_CL_MI',
        'metric_cl_fs': 'Metric_CL_FS',
        'teen_ce':'TEEN_CE',
    }
    
    # Execute n_trials
    root_log_dir = args.log_dir
    trial_id = args.trial_id
    with open(args.cfg_file, 'rb') as f :
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    print(f'--------------------Executing trial {trial_id+1}--------------------')
    # Create output directory
    trial_log_dir = osp.join(root_log_dir, f'trial_{trial_id+1}')
    if not osp.exists(trial_log_dir) :
        os.makedirs(trial_log_dir)
    args.log_dir = trial_log_dir
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log message format
                    filename=trial_log_dir +'/out.log',  # Specify the filename for log file
                    filemode='a')  # Set file mode to write ('w' will overwrite the file if it exists)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    # Create learner
    learner = getattr(importlib.import_module('.' + cfg.increm.learner.type, package='learners'),
                learners_dict[cfg.increm.learner.type])(cfg, cfg_data, args, is_train, is_distributed, n_gpus)

    if is_train :
        # Train 
        learner.train(n_trial=trial_id)

    else:
        # Evaluate
        learner.evaluate(n_trial=trial_id)
    
    # if is_train == False, summarize the metrics of all trials
    # if is_train == False :
    #     print('--------------------Summarizing results--------------------')
    #     utils.summarize_results(root_log_dir, cfg.n_trials, cfg.increm.max_task)


if __name__ == '__main__' :
    main()