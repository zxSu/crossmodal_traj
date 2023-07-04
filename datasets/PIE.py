import os

import json
import pickle as pkl
import numpy as np
from PIL import Image
import math
import torch
from torch.utils import data
from torchvision import transforms
#from torchvision.models import resnet18, mobilenet_v2

from bitrap.structures.trajectory_ops import * 
from bitrap.utils.box_utils import signedIOU
from datasets.PIE_origin import PIE
from torchvision.transforms import functional as F
import copy
from bitrap.utils.dataset_utils import bbox_to_goal_map, squarify, img_pad
import glob
import time
import pdb

import pickle

import datasets.image_process_utils as ip_utils
from datasets.MEBOW_utils import create_MEBOW_model

import matplotlib.pyplot as plt
import cv2

import torch.nn as nn


#### for convLstm stream, we can directly use the result features of the 'PIEpredict' code 



class PIEDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.root = cfg.DATASET.ROOT
        self.cfg = cfg
        # NOTE: add downsample function
        self.downsample_step = int(30/self.cfg.DATASET.FPS)
        traj_data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

        traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': [],
                       'prediction_type': ['bbox'],
                       'img_feat_save_root': self.cfg.DATASET.IMG_FEAT_SAVE_ROOT,
                       'body_ori_save_root': self.cfg.DATASET.BODY_ORI_SAVE_ROOT,
                       'pose_feat_save_root': self.cfg.DATASET.POSE_FEAT_SAVE_ROOT
                       }
        imdb = PIE(data_path=self.root)
        
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        
        self.mebow_model, self.mebow_cfg = create_MEBOW_model()
        
        
        traj_model_opts['enc_input_type'].extend(['obd_speed', 'heading_angle'])
        traj_model_opts['prediction_type'].extend(['obd_speed', 'heading_angle'])
        beh_seq = imdb.generate_data_trajectory_sequence(self.split, **traj_data_opts)
        self.data = self.get_traj_data(beh_seq, **traj_model_opts)
    
    
    
    
      
    def __getitem__(self, index):
        obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
        pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        cur_image_file = self.data['obs_image_path'][index][-1]
        pred_resolution = torch.FloatTensor(self.data['pred_resolution'][index])
        
        ####
        obs_bbox_raw = torch.FloatTensor(self.data['obs_bbox_raw'][index])
        pred_bbox_raw = torch.FloatTensor(self.data['pred_bbox_raw'][index])
        obs_obd_speed = torch.FloatTensor(self.data['obs_obd_speed'][index])
        pred_obd_speed = torch.FloatTensor(self.data['pred_obd_speed'][index])
        # process 'angle'
        obs_heading_angle = torch.FloatTensor(self.data['obs_heading_angle'][index])
        pred_heading_angle = torch.FloatTensor(self.data['pred_heading_angle'][index])
        firstFrame_heading_angle = obs_heading_angle[0, :]
        obs_heading_angle = obs_heading_angle - firstFrame_heading_angle
        pred_heading_angle = pred_heading_angle - firstFrame_heading_angle
        obs_heading_angle = ((obs_heading_angle + 360.0) % 360.0) * np.pi / 180.0
        pred_heading_angle = ((pred_heading_angle + 360.0) % 360.0) * np.pi / 180.0
        #
        intention_binary = torch.FloatTensor(self.data['intention_binary'][index])
        
        #
        obs_body_ori = torch.FloatTensor(self.data['obs_body_ori'][index])
        obs_ped_look = torch.FloatTensor(self.data['obs_ped_look'][index])
        #
        obs_pose_feats = torch.FloatTensor(self.data['obs_pose_feats'][index])
        
        
        
        #
        obs_pid = self.data['obs_pid'][index]
        
        obs_img_feats_processed = None
        
        #print(index)
        
        ret = {'input_x':obs_bbox, 
               'target_y':pred_bbox, 
               'cur_image_file':cur_image_file}
        
        ret['timestep'] = int(cur_image_file.split('/')[-1].split('.')[0])
        
        ret['pred_resolution'] = pred_resolution
        
        ####
        ret['obs_bbox_raw'] = obs_bbox_raw
        ret['pred_bbox_raw'] = pred_bbox_raw
        ret['obs_obd_speed'] = obs_obd_speed
        ret['pred_obd_speed'] = pred_obd_speed
        ret['obs_heading_angle'] = obs_heading_angle
        ret['pred_heading_angle'] = pred_heading_angle
        ret['intention_binary'] = intention_binary
        #
        ret['obs_body_ori'] = obs_body_ori
        ret['obs_pose_feat'] = obs_pose_feats
        ret['obs_ped_look'] = obs_ped_look
        #
        ret['obs_img_feat'] = obs_img_feats_processed
        
        return ret
    
    
    
    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def get_traj_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:# KeyError:
                raise KeyError('Wrong data type is selected %s' % dt)
        
        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['resolution'] = dataset['resolution']
        
        #### my adding (1) 'obd_speed', 'heading_angle'
        d['obd_speed'] = dataset['obd_speed']
        d['heading_angle'] = dataset['heading_angle']
        
        #### my adding (2) 'intention_prob', 'intention_binary'
        d['intention_prob'] = dataset['intention_prob']
        d['intention_binary'] = dataset['intention_binary']
        
        #### my adding (3) 'pedestrian_looking'
        d['ped_look'] = dataset['ped_look']
        
        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                for i in range(0, len(track) - seq_length + 1, overlap_stride):
                    tracks.append(track[i:i + seq_length])
            d[k] = tracks
        
        #### my adding (4): raw bbox which is used to crop pedestrians from image.
        d['bbox_raw'] = copy.deepcopy(d['bbox'])
        
        #  Normalize tracks using FOL paper method, 
        d['bbox'] = self.convert_normalize_bboxes(d['bbox'], d['resolution'], 
                                                  self.cfg.DATASET.NORMALIZE, self.cfg.DATASET.BBOX_TYPE)
        
        return d


    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize, bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]]/2    # bbox center
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
                _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))
            
        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_traj_data(self, data, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """
        
        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': self.cfg.MODEL.INPUT_LEN,
            'predict_length': self.cfg.MODEL.PRED_LEN,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox'],
            'img_feat_save_root': 'you need give value in here',
            'body_ori_save_root': 'you need give value in here',
            'pose_feat_save_root': 'you need give value in here'
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_traj_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])
        obs_slices = {}
        pred_slices = {}
        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            # NOTE: Add downsample function
            down = self.downsample_step
            obs_slices[k].extend([d[down-1:observe_length:down] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length+down-1::down] for d in data_tracks[k]])
        
        ######## here, extract the features of cropped images
        obs_pose_feat_results, obs_body_ori_results = ip_utils.generate_img_feats_v4(obs_slices['image'], obs_slices['bbox_raw'], obs_slices['pid'], 
                                                                             opts['body_ori_save_root'], opts['pose_feat_save_root'], 
                                                                             self.data_transform, self.mebow_model, self.mebow_cfg)
        #
        obs_heading_angle = np.array(obs_slices['heading_angle'])
        pred_heading_angle = np.array(pred_slices['heading_angle'])
        obs_obd_speed = np.array(obs_slices['obd_speed'])
        pred_obd_speed = np.array(pred_slices['obd_speed'])
        
        # get the final 'intent_np'
        intent_np = np.array(pred_slices['intention_binary'])
        intent_sum = np.sum(intent_np, axis=1)
        intent_flag = np.where(intent_sum>0)[0]
        #
        intent_np_final = np.zeros([intent_np.shape[0], 1])
        intent_np_final[intent_flag] = 1
        
        #
        ret =  {'obs_image_path': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'obs_resolution': obs_slices['resolution'],
                'pred_image_path': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'pred_resolution': pred_slices['resolution'],
                'obs_bbox': np.array(obs_slices['bbox']), #enc_input,
                'pred_bbox': np.array(pred_slices['bbox']), #pred_target,
                'obs_bbox_raw': np.array(obs_slices['bbox_raw']),    # my adding
                'pred_bbox_raw': np.array(pred_slices['bbox_raw']),    # my adding
                'obs_obd_speed': obs_obd_speed,    # my adding
                'pred_obd_speed': pred_obd_speed,    # my adding
                'obs_heading_angle': obs_heading_angle,    # my adding
                'pred_heading_angle': pred_heading_angle,    # my adding
                'obs_body_ori': obs_body_ori_results,    # my adding (the raw body orientation. this is what i wanted in image space.)
                'obs_pose_feats': obs_pose_feat_results,    # my adding
                'intention_binary': intent_np_final,    # my adding. There are the same binary value between 'obs_seq' and 'pred_seq'.
                'obs_ped_look': np.array(obs_slices['ped_look']),    # my adding
                }
        
        return ret
    
    
    
    def get_path(self,
                 file_name='',
                 save_folder='models',
                 dataset='pie',
                 model_type='trajectory',
                 save_root_folder='data/'):
        """
        A path generator method for saving model and config data. It create directories if needed.
        :param file_name: The actual save file name , e.g. 'model.h5'
        :param save_folder: The name of folder containing the saved files
        :param dataset: The name of the dataset used
        :param save_root_folder: The root folder
        :return: The full path for the model name and the path to the final folder
        """
        save_path = os.path.join(save_root_folder, dataset, model_type, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path