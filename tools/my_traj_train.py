'''
'''
import pdb
import os
import sys
sys.path.append(os.path.realpath('..'))
#pdb.set_trace()

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import pickle as pkl
#from datasets.build import make_dataloader
from datasets import make_dataloader

#from bitrap.modeling.build import make_model
from bitrap.modeling import make_model

from bitrap.engine import build_engine
from bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal
from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg
from collections import OrderedDict
import pdb

import visdom

import random


def build_optimizer(cfg, model):
    all_params = model.parameters()
    # optimizer = optim.RMSprop(all_params, lr=cfg.SOLVER.LR)
    optimizer = optim.Adam(all_params, lr=cfg.SOLVER.LR)
    return optimizer



def build_optimizer_v2(cfg, model, multitask_weights):
    model_params = model.parameters()
    all_params = list(model_params)
    all_params.append(multitask_weights)
    # optimizer = optim.RMSprop(all_params, lr=cfg.SOLVER.LR)
    optimizer = optim.Adam(all_params, lr=cfg.SOLVER.LR)
    return optimizer


######## 2021/07/13
def get_dtypes(use_gpu=1):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

######## 2021/07/13
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.normal_(m.weight, 0.0, 0.02)

######## 2021/07/13
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



######## 2021/08/17
def get_and_save_random_state(state_save_path):
    curr_np_random_state = np.random.get_state()
    curr_torch_random_state = torch.random.get_rng_state().numpy()
    final_tuple = (curr_np_random_state, curr_torch_random_state)
    with open(state_save_path, 'wb') as f:
        pkl.dump(final_tuple, f)
    #
    #torch.backends.cudnn.benchmark = False    #
    torch.backends.cudnn.deterministic = True


def set_random_state(state_save_path):
    with open(state_save_path, 'rb') as f:
        tuple_load = pkl.load(f)
    np_random_state = tuple_load[0]
    torch_random_state = torch.tensor(tuple_load[1])
    # set
    np.random.set_state(np_random_state)
    torch.random.set_rng_state(torch_random_state)
    #torch.backends.cudnn.benchmark = False    #
    torch.backends.cudnn.deterministic = True
    print('load random state from: '+state_save_path)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_multimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_unimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/acl_unimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/concat_unimodal_PIE.yml",
        default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_multimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_unimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/acl_unimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/concat_unimodal_JAAD.yml",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--decoder_type",
        help="Modify decoder type using the command-line",
        default='bezier_curve',
        #default="mlp",
        #default="gru",
        type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    
    
    
    ######## I have to say that random states are very important. 
    ######## Sometimes, training our model with a random state can achieve good performance (for example, these results reported in our paper)
    # # set fixed random state (optional)
    # #random_state_save_path = '/home/suzx/eclipse-workspace/crossmodal_traj/random_state/random_state_unimodal_JAAD.pkl'
    # #random_state_save_path = '/home/suzx/eclipse-workspace/crossmodal_traj/random_state/random_state_unimodal_PIE.pkl'
    # #random_state_save_path = '/home/suzx/eclipse-workspace/crossmodal_traj/random_state/random_state_multimodal_JAAD.pkl'
    # random_state_save_path = '/home/suzx/eclipse-workspace/crossmodal_traj/random_state/random_state_multimodal_PIE.pkl'
    # get_and_save_random_state(random_state_save_path)
    # #set_random_state(random_state_save_path)
    
    
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    #
    cfg.MODEL.DECODER_TYPE = args.decoder_type
    
    # build model, optimizer and scheduler
    model = make_model(cfg)
    model = model.to(cfg.DEVICE)
    
    #model.apply(init_weights)    # init the weight of model (optional)
    
    optimizer = build_optimizer(cfg, model)
    
    print('optimizer built!')
    # NOTE: add separate optimizers to train single object predictor and interaction predictor
    
    if cfg.USE_WANDB:
        logger = Logger("FOL",
                        cfg,
                        project = cfg.PROJECT,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger("FOL")

    dataloader_params ={
            "batch_size": cfg.SOLVER.BATCH_SIZE,
            "shuffle": True,
            "num_workers": cfg.DATALOADER.NUM_WORKERS
            }
    
    # get dataloaders
    train_dataloader = make_dataloader(cfg, 'train')
    val_dataloader = make_dataloader(cfg, 'val')
    test_dataloader = make_dataloader(cfg, 'test')
    
    
    print('Dataloader built!')
    # get train_val_test engines
    do_train, do_val, inference = build_engine(cfg)
    print('Training engine built!')
    
    #
    if (cfg.METHOD=='TransPredict_Unimodal') and (cfg.MODEL.USE_MODALITY_ATTEN==False):
        config_name = 'Concat_Unimodal'
    elif (cfg.METHOD=='TransPredict_Multimodal') and (cfg.MODEL.USE_MODALITY_ATTEN==False):
        config_name = 'Concat_Multimodal'
    else:
        config_name = cfg.METHOD

    save_checkpoint_dir = os.path.join(cfg.CKPT_DIR, config_name, cfg.MODEL.DECODER_TYPE)
    if cfg.MODEL.LATENT_DIM>0:
        save_checkpoint_dir = os.path.join(save_checkpoint_dir, 'latent_'+str(cfg.MODEL.LATENT_DIM))
    if not os.path.exists(save_checkpoint_dir):
        os.makedirs(save_checkpoint_dir)
    
    # NOTE: hyperparameter scheduler
    model.param_scheduler = ParamScheduler()
    model.param_scheduler.create_new_scheduler(
                                        name='kld_weight',
                                        annealer=sigmoid_anneal,
                                        annealer_kws={
                                            'device': cfg.DEVICE,
                                            'start': 0,
                                            'finish': 100.0,
                                            'center_step': 400.0,
                                            'steps_lo_to_hi': 100.0, 
                                        })
    
    model.param_scheduler.create_new_scheduler(
                                        name='z_logit_clip',
                                        annealer=sigmoid_anneal,
                                        annealer_kws={
                                            'device': cfg.DEVICE,
                                            'start': 0.05,
                                            'finish': 5.0, 
                                            'center_step': 300.0,
                                            'steps_lo_to_hi': 300.0 / 5.
                                        })
    
    
    if cfg.SOLVER.scheduler == 'exp':
        # exponential schedule
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.scheduler == 'plateau':
        # Plateau scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
                                                            min_lr=1e-07, verbose=1)    # the original 'factor' is 0.2
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)
                                                        
    print('Schedulers built!')
    
    #
    test_frequency = 1    #
    #
    best_val_loss = 1000
    best_acc = 0
    best_f1 = 0
    best_c_ade = 10000
    best_c_fde = 10000
    
    # visdom
    viz = visdom.Visdom()
    #
    viz.line([0.0], [0.0], win='ci_train', opts=dict(title='ci_train'))
    viz.line([0.0], [0.0], win='traj_train', opts=dict(title='traj_train'))
    viz.line([0.0], [0.0], win='ego_train', opts=dict(title='ego_train'))
    viz.line([0.0], [0.0], win='ego_speed_train', opts=dict(title='ego_speed_train'))
    viz.line([0.0], [0.0], win='ego_angle_train', opts=dict(title='ego_angle_train'))
    viz.line([[0.0, 0.0, 0.0]], [0.0], win='task_weight', opts=dict(title='task_weight', legend=['ci', 'traj', 'ego']))
    #
    viz.line([0.0], [0.0], win='ci_val', opts=dict(title='ci_val'))
    #
    viz.line([[0.0, 0.0]], [0.0], win='ci_inference', opts=dict(title='ci_inference'))
    viz.line([[0.0, 0.0]], [0.0], win='c_ade_fde_inference', opts=dict(title='c_ade_fde_inference'))
    viz.line([[0.0, 0.0, 0.0]], [0.0], win='ade_inference', opts=dict(title='ade_inference'))
    t_train = 0
    
    
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        #logger.info("Epoch:{}, training now".format(epoch))
        print("Epoch:{}, training now, kld_weight:{}".format(epoch, model.param_scheduler.kld_weight))
        t_train = do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.DEVICE, logger=logger, lr_scheduler=lr_scheduler, visdom_viz=viz, t=t_train, 
                           consider_crossIntent=cfg.USE_CROSSING_LOSS, dataset_name=cfg.DATASET.NAME)
        print("Epoch:{}, validating now".format(epoch))
        val_loss = do_val(cfg, epoch, model, val_dataloader, cfg.DEVICE, logger=logger, visdom_viz=viz, t=epoch, dataset_name=cfg.DATASET.NAME)
        if (epoch+1) % test_frequency == 0:
            print("Epoch:{}, inferencing now".format(epoch))
            acc, f1, c_ade, c_fde = inference(cfg, epoch, model, test_dataloader, cfg.DEVICE, logger=logger, visdom_viz=viz, t=epoch, dataset_name=cfg.DATASET.NAME)
            # my adding
            #if acc>best_acc:
            if c_ade<best_c_ade:
                torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'BestModel_Inference.pth'.format(str(epoch).zfill(3))))
                best_acc = acc
                best_f1 = f1
                best_c_ade = c_ade
                best_c_fde = c_fde
            
        #torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))
        
        # my adding
        if val_loss<best_val_loss:
            torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'BestModel_Val.pth'.format(str(epoch).zfill(3))))
            best_val_loss = val_loss

        # update LR
        if cfg.SOLVER.scheduler != 'exp':
            lr_scheduler.step(val_loss)
        
if __name__ == '__main__':
    main()



