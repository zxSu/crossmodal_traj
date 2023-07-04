import os
import sys
sys.path.append(os.path.realpath('.'))

import torch
from torch import nn, optim
from torch.nn import functional as F

import pickle as pkl
from datasets import make_dataloader
from bitrap.modeling import make_model
from bitrap.engine import build_engine


from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg
from termcolor import colored 


def main(cfg):
    # build model, optimizer and scheduler
    model = make_model(cfg)
    model = model.to(cfg.DEVICE)
    
    if os.path.isfile(cfg.CKPT_DIR):
        model.load_state_dict(torch.load(cfg.CKPT_DIR))
        print(colored('Loaded checkpoint:{}'.format(cfg.CKPT_DIR), 'blue', 'on_green'))
    else:
        print(colored('The cfg.CKPT_DIR id not a file: {}'.format(cfg.CKPT_DIR), 'green', 'on_red'))
    
    if cfg.USE_WANDB:
        logger = Logger("MPED_RNN",
                        cfg,
                        project = cfg.PROJECT,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger("MPED_RNN")
    
    # get dataloaders
    test_dataloader = make_dataloader(cfg, 'test')
    
    if hasattr(logger, 'run_id'):
        run_id = logger.run_id
    else:
        run_id = 'no_wandb'
    _, _, inference = build_engine(cfg)
    
    acc, f1, c_ade, c_fde = inference(cfg, None, model, test_dataloader, cfg.DEVICE, logger=logger, test_mode=False, visdom_viz=None, t=None, 
                                      dataset_name=cfg.DATASET.NAME, traj_viz=False, eval_kde_nll=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--config_file",
        default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_unimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_unimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_multimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/my_trans_multimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/acl_unimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/concat_unimodal_PIE.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/acl_unimodal_JAAD.yml",
        #default="/home/suzx/eclipse-workspace/crossmodal_traj/configs/concat_unimodal_JAAD.yml",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--decoder_type",
        help="Modify decoder type using the command-line",
        default="bezier_curve",
        #default="mlp",
        #default="gru",
        type=str
    )
    parser.add_argument(
        "--ckpt_root",
        help="root to all checkpoints",
        default="/home/suzx/Desktop/jaad_pie_checkpoints_final",
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
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    # related to 'ablation study'
    cfg.MODEL.DECODER_TYPE = args.decoder_type
    
    
    #
    if (cfg.METHOD=='TransPredict_Unimodal') and (cfg.MODEL.USE_MODALITY_ATTEN==False):
        config_name = 'Concat_Unimodal'
    else:
        config_name = cfg.METHOD
    
    if cfg.MODEL.LATENT_DIM==0:
        #ckpt_path = os.path.join(args.ckpt_root, cfg.CKPT_DIR, config_name, cfg.MODEL.DECODER_TYPE, 'BestModel_Inference.pth')
        ckpt_path = os.path.join(args.ckpt_root, cfg.CKPT_DIR, config_name, cfg.MODEL.DECODER_TYPE, 'BestModel_Val.pth')
    else:
        #ckpt_path = os.path.join(args.ckpt_root, cfg.CKPT_DIR, config_name, cfg.MODEL.DECODER_TYPE, 'latent_'+str(cfg.MODEL.LATENT_DIM), 'BestModel_Inference.pth')
        ckpt_path = os.path.join(args.ckpt_root, cfg.CKPT_DIR, config_name, cfg.MODEL.DECODER_TYPE, 'latent_'+str(cfg.MODEL.LATENT_DIM), 'BestModel_Val.pth')
    
    cfg.CKPT_DIR = ckpt_path

    main(cfg)



