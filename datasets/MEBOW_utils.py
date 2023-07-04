# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import cv2
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import DepthLoss
from core.loss import hoe_diff_loss
from core.loss import Bone_loss

from core.function import train
from core.function import validate, my_demo

from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models

from core.evaluate import accuracy
from core.evaluate import comp_deg_error, continous_comp_deg_error, draw_orientation, ori_numpy, draw_pred_orientation



my_avgPool_op = nn.AvgPool2d((4, 4), stride=(4, 4))



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args





def create_MEBOW_model():
    args = parse_args()
    
    args.cfg = '/home/suzx/eclipse-workspace/crossmodal_traj/MEBOW/experiments/coco/my_cfg.yaml'
    
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    return model, cfg




# my adding
def box2cs(box, aspect_ratio, pixel_std):
    x, y, w, h = box[:4]
    #
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale



def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result



def mebow_evaluate(cfg, img_np_list, img_path_list, transform, model, output_dir):
    
    #
    img_width = cfg.MODEL.IMAGE_SIZE[0]
    img_height = cfg.MODEL.IMAGE_SIZE[1]
    aspect_ratio = img_width * 1.0 / img_height
    pixel_std = 200
    
    # switch to evaluate mode
    model.eval()
    
    ori_result_list = []
    pose_result_list = []
    with torch.no_grad():
        
        # (1) load image
        for (curr_input_np, curr_path) in zip(img_np_list, img_path_list):
            
            curr_input_original = copy.deepcopy(curr_input_np)
            
            # (3.1) wrap affine (this step is very important)
            h_img = curr_input_np.shape[0]
            w_img = curr_input_np.shape[1]
            bbox = [0, 0, w_img, h_img]
            center, scale = box2cs(bbox, aspect_ratio, pixel_std)
            trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
            curr_input_np = cv2.warpAffine(
                curr_input_np,
                trans,
                (int(img_width), int(img_height)),
                flags=cv2.INTER_LINEAR)
            
            # (3.2) resize
            #curr_input_np = cv2.resize(curr_input_np, (img_width, img_height))
            
            # (4) extra transform
            curr_input_tensor = transform(curr_input_np)
            curr_input_tensor = curr_input_tensor.unsqueeze(dim=0).cuda()
            
            # (5) network inference
            # compute output
            plane_output, hoe_output = model(curr_input_tensor)
            
            # output to pred_degree
            index_degree = hoe_output.argmax(axis = 1)
            pred_ori = index_degree * 5
            pred_ori = pred_ori.detach().cpu().numpy()
            
            # for our work, the 'plane_output' is too big (17x64x48), we can resize it to a smaller size.
            plane_output = my_avgPool_op(plane_output)
            plane_output = plane_output.detach().cpu().numpy()
            
            #
            ori_result_list.append(pred_ori[0])
            pose_result_list.append(plane_output[0, :])
    
    return ori_result_list, pose_result_list






