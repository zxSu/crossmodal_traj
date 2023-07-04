import sys
import numpy as np
import copy
import math
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
import torch.distributions as td

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

from .my_net_utils import *

from bitrap.modeling.conv_lstm import ConvLSTM, CLSTM, CGRU


from bitrap.modeling.transformer_utils.transformer_encoder import TransformerEncoder, fill_with_neg_inf
from bitrap.modeling.transformer_utils.position_encoding import PositionalEncoding









######## 2021/05/17 suzx
# multihead attention! (maybe better then single-head attention)
class ACL_Transformer_Encoder(nn.Module):
    
    def __init__(self, feat_dim=272):
        super(ACL_Transformer_Encoder, self).__init__()
        
        #
        self.num_heads = 4
        self.layers = 3
        self.attn_dropout = 0.0    # 0.1
        self.relu_dropout = 0.0    # 0.1
        self.res_dropout = 0.0    # 0.1
        self.embed_dropout = 0.0    # 0.25
        self.attn_mask = False
        
        # 2. Crossmodal Attentions
        self.trans_ego_with_pose = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        self.trans_bbox_with_pose = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        #
        self.trans_ego_with_bbox = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        self.trans_pose_with_bbox = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        #
        self.trans_bbox_with_ego = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        self.trans_pose_with_ego = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        
        # 3. self attention
        self.trans_pose = self.get_transformer_network(embed_dim=feat_dim*2, attn_dropout=self.attn_dropout)
        self.trans_bbox = self.get_transformer_network(embed_dim=feat_dim*2, attn_dropout=self.attn_dropout)
        self.trans_ego = self.get_transformer_network(embed_dim=feat_dim*2, attn_dropout=self.attn_dropout)
    
    
    def get_transformer_network(self, embed_dim, attn_dropout, layers=-1):
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    
    def forward(self, x_ego, x_pose, x_bbox):
        """
        ego, pose (pedestrian), and bbox should have dimension [batch_size, seq_len, n_features]
        """
        
        x_ego = x_ego.transpose(0, 1)
        x_pose = x_pose.transpose(0, 1)
        x_bbox = x_bbox.transpose(0, 1)
        
        # (pose, bbox) --> ego
        h_ego_with_pose, _ = self.trans_ego_with_pose(x_ego, x_pose, x_pose)    # Dimension (L, N, d_l)
        h_ego_with_bbox, _ = self.trans_ego_with_bbox(x_ego, x_bbox, x_bbox)    # Dimension (L, N, d_l)
        # (ego, bbox) --> pose
        h_pose_with_ego, _ = self.trans_pose_with_ego(x_pose, x_ego, x_ego)    # Dimension (L, N, d_l)
        h_pose_with_bbox, _ = self.trans_pose_with_bbox(x_pose, x_bbox, x_bbox)    # Dimension (L, N, d_l)
        # (ego, pose) --> bbox
        h_bbox_with_ego, _ = self.trans_bbox_with_ego(x_bbox, x_ego, x_ego)    # Dimension (L, N, d_l)
        h_bbox_with_pose, _ = self.trans_bbox_with_pose(x_bbox, x_pose, x_pose)    # Dimension (L, N, d_l)
        
        # concat
        h_ego_raw = torch.cat([h_ego_with_pose, h_ego_with_bbox], dim=-1)
        h_pose_raw = torch.cat([h_pose_with_ego, h_pose_with_bbox], dim=-1)
        h_bbox_raw = torch.cat([h_bbox_with_ego, h_bbox_with_pose], dim=-1)
        
        # ego --> ego
        h_ego, _ = self.trans_ego(h_ego_raw)
        # pose --> pose
        h_pose, _ = self.trans_pose(h_pose_raw)
        # bbox --> bbox
        h_bbox, _ = self.trans_bbox(h_bbox_raw)
        
        #
        h_ego = h_ego.transpose(0, 1)
        h_pose = h_pose.transpose(0, 1)
        h_bbox = h_bbox.transpose(0, 1)
        h_final = torch.cat([h_ego[:, -1], h_pose[:, -1], h_bbox[:, -1]], dim=-1)
        
        return h_final









########


class ACL_Predictor_Unimodal(nn.Module):
     
    def __init__(self, cfg, dataset_name='', use_img_feat=True, use_attrib_feat=True, 
                 use_cross_atten=True):
        super(ACL_Predictor_Unimodal, self).__init__()
         
        self.cfg = copy.deepcopy(cfg)
        self.pred_len = self.cfg.PRED_LEN
        self.param_scheduler = None    # this variable will be set in 'train.py'
        self.dataset_name = dataset_name
         
        #
        self.use_img_feat = use_img_feat
        self.use_attrib_feat = use_attrib_feat
        self.use_cross_atten = use_cross_atten
         
        #
        dim_in_concat = 0
         
         
        ########
        if self.use_cross_atten:
            self.encoder = ACL_Transformer_Encoder(feat_dim=self.cfg.INPUT_EMBED_SIZE)
            # we assume all modalities will be used
            self.use_img_feat = True
        else:
            self.encoder = None
         
         
        ########
        if self.use_img_feat:
            #### v1
            #### img encoder (conv-lstm)
            dim_unit = 17
            input_shape = [16, 12]
            self.img_encoder = CLSTM(shape=input_shape, input_channels=dim_unit, filter_size=3, num_features=dim_unit)    # 'num_features' means the channels for each gate.
               
            # oops, the extracted feature cube is still too big, so another conv-net is needed.
            self.post_conv = nn.Sequential(
                nn.Conv2d(dim_unit, dim_unit*2, 3, 1, 1),
                #nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
                nn.BatchNorm2d(dim_unit*2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(dim_unit*2, dim_unit*4, 3, 1, 1),
                nn.BatchNorm2d(dim_unit*4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(dim_unit*4, dim_unit*8, 3, 1, 1),
                nn.BatchNorm2d(dim_unit*8),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
             
         
         
        ########
        self.bbox_embed_1 = make_mlp(dim_list=[4, self.cfg.INPUT_EMBED_SIZE], activation='relu', batch_norm=False, bias=True, dropout=0)
        self.bbox_encoder = nn.Conv1d(self.cfg.INPUT_EMBED_SIZE, self.cfg.INPUT_EMBED_SIZE, kernel_size=1, padding=0, bias=False)
        
         
        ########
        dim_in_ego = 2
        if self.dataset_name=='JAAD':
            dim_in_ego = 1
        self.ego_embed = make_mlp(dim_list=[dim_in_ego, self.cfg.INPUT_EMBED_SIZE], activation='relu', batch_norm=False, bias=True, dropout=0)    # speed & angle
        self.ego_encoder = nn.Conv1d(self.cfg.INPUT_EMBED_SIZE, self.cfg.INPUT_EMBED_SIZE, kernel_size=1, padding=0, bias=False)
         
         
        ########
        if self.use_cross_atten:
            dim_in_concat = self.cfg.ENC_HIDDEN_SIZE * 6
        
        
        
        
        if self.use_attrib_feat:
            if self.cfg.ATTRIB_FEAT_TYPE=='simple':
                if self.dataset_name=='JAAD':
                    self.dim_attrib = 7    # 6 or 7
                else:
                    self.dim_attrib = 8    # 6 or 8
                dim_social = 128
                self.social_layer_simple = make_mlp(dim_list=[self.dim_attrib, 64, dim_social], activation='relu', batch_norm=False, bias=True, dropout=0)
            else:
                print('do not support complex social modeling.')
                assert(1==0)
                 
            #
            dim_in_concat += dim_social
        
         
         
         
         
         
        dim_in_final = dim_in_concat
         
         
        #
        self.intention_predictor = make_mlp(dim_list=[dim_in_final, int(dim_in_final/2), 1], activation='', batch_norm=False, bias=True, dropout=0)    # the loss i should use is: logistic loss (binary cross entropy)
        
        self.intention_predictor_v2 = nn.Sequential(nn.Linear(dim_in_final, int(dim_in_final/2)), 
                                                    nn.ReLU(), 
                                                    nn.Linear(int(dim_in_final/2), 64), 
                                                    nn.ReLU(), 
                                                    nn.Linear(64, self.cfg.INTENT_OUT_DIM)
                                                    )
        
        
         
        ####
        self.control_points_mlp_predictor = nn.Sequential(nn.Linear(dim_in_final, 512), 
                                                          nn.ReLU(), 
                                                          nn.Linear(512, 256), 
                                                          nn.ReLU(), 
                                                          nn.Linear(256, 12)
                                                          )
         
     
    def obtain_feat_individual(self, img_inputs, obs_bbox=None, obs_ego=None):
        
        ########
        if self.use_img_feat:
            #### v1
            #### (1) img feature
            feats_img_encode_raw, _ = self.img_encoder(img_inputs)    # oops, cuda out of memory. the image feature sequence is too large!
            #
            feats_img_encode_raw = feats_img_encode_raw.reshape(-1, img_inputs.shape[2], img_inputs.shape[3], img_inputs.shape[4])
            feats_img_encode = self.post_conv(feats_img_encode_raw)
            feats_img_encode = feats_img_encode.reshape(img_inputs.shape[0], img_inputs.shape[1], -1)
         
        
        ########
        embed_bbox_encode = self.bbox_embed_1(obs_bbox)
        embed_bbox_encode = embed_bbox_encode.transpose(1, 2)
        feats_bbox_encode = self.bbox_encoder(embed_bbox_encode)
        feats_bbox_encode = feats_bbox_encode.transpose(1, 2)
         
        ########
        embed_ego_encode = self.ego_embed(obs_ego)
        embed_ego_encode = embed_ego_encode.transpose(1, 2)
        feats_ego_encode = self.ego_encoder(embed_ego_encode)
        feats_ego_encode = feats_ego_encode.transpose(1, 2)
        
        ########
        if self.use_cross_atten:
            #### cross transformer
            final_feat = self.encoder(feats_ego_encode, feats_img_encode, feats_bbox_encode)
        else:
            final_feat = self.encoder(embed_bbox_encode)
         
         
         
        return final_feat
     
    
     
     
    ########
    def obtain_feat_social_simple(self, obsLast_look, obsLast_body_ori, obsLast_bbox, obsLast_ego_act):
        social_attrib = torch.cat([obsLast_look, obsLast_body_ori, obsLast_bbox, obsLast_ego_act], dim=1)
        social_feats = self.social_layer_simple(social_attrib)
         
        return social_feats
     
     
     
    ######## the simple gru decoder (predict the goal, then make interpolation)
    def traj_predict(self, obsLast_bbox, bbox_h):
        
        # directly predict the 'control points'
        pred_control_points = self.control_points_mlp_predictor(bbox_h)
        pred_control_points = pred_control_points.reshape([-1, 3, 4])
        #
        control_cx_cy = pred_control_points[:, :, :2]
        control_w_h = pred_control_points[:, :, 2:4]
        # bezier curve
        t_linspace = (torch.linspace(1, self.pred_len, steps=self.pred_len) / self.pred_len).repeat(bbox_h.shape[0], 1).cuda()
        curve_cx = obsLast_bbox[:, 0].reshape([-1, 1]) * (1-t_linspace)**3 + 3 * control_cx_cy[:, 0, 0].reshape([-1, 1]) * t_linspace * (1-t_linspace)**2 + 3 * control_cx_cy[:, 1, 0].reshape([-1, 1]) * t_linspace**2 * (1-t_linspace) + control_cx_cy[:, 2, 0].reshape([-1, 1]) * t_linspace**3
        curve_cy = obsLast_bbox[:, 1].reshape([-1, 1]) * (1-t_linspace)**3 + 3 * control_cx_cy[:, 0, 1].reshape([-1, 1]) * t_linspace * (1-t_linspace)**2 + 3 * control_cx_cy[:, 1, 1].reshape([-1, 1]) * t_linspace**2 * (1-t_linspace) + control_cx_cy[:, 2, 1].reshape([-1, 1]) * t_linspace**3
        #
        curve_w = obsLast_bbox[:, 2].reshape([-1, 1]) * (1-t_linspace)**3 + 3 * control_w_h[:, 0, 0].reshape([-1, 1]) * t_linspace * (1-t_linspace)**2 + 3 * control_w_h[:, 1, 0].reshape([-1, 1]) * t_linspace**2 * (1-t_linspace) + control_w_h[:, 2, 0].reshape([-1, 1]) * t_linspace**3
        curve_h = obsLast_bbox[:, 3].reshape([-1, 1]) * (1-t_linspace)**3 + 3 * control_w_h[:, 0, 1].reshape([-1, 1]) * t_linspace * (1-t_linspace)**2 + 3 * control_w_h[:, 1, 1].reshape([-1, 1]) * t_linspace**2 * (1-t_linspace) + control_w_h[:, 2, 1].reshape([-1, 1]) * t_linspace**3
            
        pred_traj = torch.stack([curve_cx, curve_cy, curve_w, curve_h], dim=2)
        
        return pred_traj
     
    
     
     
     
     
     
     
     
    def forward(self, img_inputs, 
                look_inputs, body_ori_inputs, 
                obs_bbox=None, obs_ego=None):
         
        #
        past_h = self.obtain_feat_individual(img_inputs, obs_bbox, obs_ego)
        
        
        # social features
        if self.use_attrib_feat:
            if self.cfg.ATTRIB_FEAT_TYPE=='simple':
                social_feat = self.obtain_feat_social_simple(look_inputs, body_ori_inputs, obs_bbox[:, -1, :], obs_ego[:, -1])
            else:
                social_feat = None
                print('do not support complex social modeling.')
                assert(1==0)
            #
            feat_obs = torch.cat([past_h, social_feat], dim=1)
        else:
            feat_obs = past_h
         
         
        # decoding (unimodal prediction)
        pred_traj = self.traj_predict(obs_bbox[:, -1, :], feat_obs)
        
        crossing_outputs = self.intention_predictor_v2(feat_obs)
         
        return crossing_outputs, pred_traj
     
     
     
     
    def predict(self, img_inputs, 
                look_inputs, body_ori_inputs, 
                obs_bbox=None, obs_ego=None):
         
        #
        past_h = self.obtain_feat_individual(img_inputs, obs_bbox, obs_ego)
        
        
        # social features
        if self.use_attrib_feat:
            if self.cfg.ATTRIB_FEAT_TYPE=='simple':
                social_feat = self.obtain_feat_social_simple(look_inputs, body_ori_inputs, obs_bbox[:, -1, :], obs_ego[:, -1])
            else:
                social_feat = None
                print('do not support complex social modeling.')
                assert(1==0)
            #
            feat_obs = torch.cat([past_h, social_feat], dim=1)
        else:
            feat_obs = past_h
         
         
        # decoding (unimodal prediction)
        pred_traj = self.traj_predict(obs_bbox[:, -1, :], feat_obs)
        
        prob_cross_softmax = F.softmax(self.intention_predictor_v2(feat_obs), dim=1)
        pred_cross = prob_cross_softmax.max(dim=1)[1]
        pred_cross = pred_cross.reshape([-1, 1])
         
        return pred_cross, pred_traj





