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

from .my_net_utils import *

from bitrap.modeling.conv_lstm import ConvLSTM, CLSTM, CGRU


from bitrap.modeling.transformer_utils.transformer_encoder import TransformerEncoder, fill_with_neg_inf
from bitrap.modeling.transformer_utils.position_encoding import PositionalEncoding




######## related to mask for self-attention in transformer decoding
def my_buffered_future_mask(dim1, dim2, use_gpu=True):
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if use_gpu:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]




def update_pred_result_regressive(old_pred_result, new_pred_result):
    new_pred_result = new_pred_result.unsqueeze(dim=1)
    pred_result_regressive = torch.cat([old_pred_result, new_pred_result], dim=1)
    return pred_result_regressive





######## 2021/05/25 suzx
class Transformer_Decoder(nn.Module):
    
    def __init__(self, embed_dim=272, nhead=4, num_layers=3):
        super(Transformer_Decoder, self).__init__()
        
        self.embed_scale = math.sqrt(embed_dim)
        
        ######## use the offical implementation of transformer decoder in pytorch
        # oops, do not forget the 'positional encoding'
        self.embed_positions = PositionalEncoding(embed_dim)
        #
        bbox_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.bbox_decoder = nn.TransformerDecoder(bbox_decoder_layer, num_layers=num_layers)
    
    
    #### used in training
    def forward(self, embeds_target, memory, tgt_mask):
        # scale embedding
        embeds_target = embeds_target * self.embed_scale
        # add 'positional encoding', then 'dropout'
        x = self.embed_positions(embeds_target.transpose(0, 1)).transpose(0, 1)   # Add positional encoding
        x = F.dropout(x, p=0.1, training=self.training)
        #
        pred_feats_curr = self.bbox_decoder(x, memory, tgt_mask=tgt_mask)
        
        return pred_feats_curr







######## gaussian latent net (2021/04/16)
class Gaussian_Latent_Net(nn.Module):
    
    def __init__(self, dim_pastFeat=512, dim_embed=64, dim_state=64, dim_in_ego=2, use_gpu=1, 
                 manner_construct_future_h='concat', latent_dim=16):
        super(Gaussian_Latent_Net, self).__init__()
        
        self.use_gpu = use_gpu
        #self.latent_dim = int(dim_pastFeat / 16)    # dim_pastFeat / 8
        self.latent_dim = latent_dim
        #
        self.manner_construct_future_h = manner_construct_future_h
        
        dim_embed_bbox = dim_embed
        dim_embed_ego = dim_embed
        dim_trans = dim_embed
        
        #
        if self.manner_construct_future_h=='self_atten':
            self.future_trans_bbox = None
        else:
            # 'cross_atten'
            self.future_trans_ego_with_bbox = TransformerEncoder(embed_dim=dim_trans, 
                                                                 num_heads=4, 
                                                                 layers=3, 
                                                                 attn_dropout=0.0,    # 0.1
                                                                 relu_dropout=0.0,    # 0.1
                                                                 res_dropout=0.0,    # 0.1
                                                                 embed_dropout=0.0,    # 0.25
                                                                 attn_mask=False)
        
        #
        self.p_z_x = nn.Sequential(nn.Linear(dim_pastFeat, 256), 
                                   nn.ReLU(), 
                                   nn.Linear(256, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128, self.latent_dim*2)
                                   )
        self.q_z_xy = nn.Sequential(nn.Linear(dim_pastFeat+dim_state, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.latent_dim*2)
                                    )
        
        #
        self.bbox_embed = make_embedding(dim_in=4, dim_out=dim_embed_bbox, activation='relu', batch_norm=False, bias=True, dropout=0)
        self.bbox_encoder = nn.Conv1d(dim_embed_bbox, dim_embed_bbox, kernel_size=1, padding=0, bias=False)
        
        #
        self.ego_embed = make_embedding(dim_in=dim_in_ego, dim_out=dim_embed_ego, activation='relu', batch_norm=False, bias=True, dropout=0)
        self.ego_encoder = nn.Conv1d(dim_embed_ego, dim_embed_ego, kernel_size=1, padding=0, bias=False)
    
    
    def forward(self, past_h, future_bboxs, future_ego, num_samples_forEachTraj=1):
        
        z_mu_logvar_p = self.p_z_x(past_h)
        z_mu_p = z_mu_logvar_p[:, :self.latent_dim]
        z_logvar_p = z_mu_logvar_p[:, self.latent_dim:]
        
        if future_bboxs is not None:
            
            # future traj encoding
            future_embeds_bbox = self.bbox_embed(future_bboxs)
            future_embeds_bbox = future_embeds_bbox.transpose(1, 2)
            feats_bbox_future = self.bbox_encoder(future_embeds_bbox)
            feats_bbox_future = feats_bbox_future.transpose(1, 2)
            
            # future ego encoding
            future_embeds_ego = self.ego_embed(future_ego)
            future_embeds_ego = future_embeds_ego.transpose(1, 2)
            feats_ego_future = self.ego_encoder(future_embeds_ego)
            feats_ego_future = feats_ego_future.transpose(1, 2)

            
            ######## how to represent the 'future_h'
            #
            if self.manner_construct_future_h=='cross_atten':
                # cross transformer between 'ego' and 'bbox'
                y_ego = feats_ego_future.transpose(0, 1)
                y_bbox = feats_bbox_future.transpose(0, 1)
                future_h_s, _ = self.future_trans_ego_with_bbox(y_ego, y_bbox, y_bbox)
                future_h_s = future_h_s.transpose(0, 1)
                # pool among all time steps
                future_h = future_h_s.max(dim=1)[0]
                #future_h = torch.mean(future_h_s, dim=1)
            else:
                future_h = None
            
            #
            q_in = torch.cat([past_h, future_h], dim=-1)
            z_mu_logvar_q = self.q_z_xy(q_in)
            z_mu_q = z_mu_logvar_q[:, :self.latent_dim]
            z_logvar_q = z_mu_logvar_q[:, self.latent_dim:]
            
            # compute KL Divergence between two gaussian distribution
            # KL(q_z_xy||p_z_x). 'q_z_xy' is the goal, while 'p_z_x' is the approximation
            # based on the equations from 'https://zhuanlan.zhihu.com/p/55778595'
            term_1 = z_logvar_q - z_logvar_p
            term_2 = z_logvar_q.exp() / z_logvar_p.exp()
            term_3 = (z_mu_q - z_mu_p).pow(2) / z_logvar_p.exp()
            kld = -0.5 * (term_1 - term_2 - term_3 + 1)
            kld = kld.sum(dim=-1).mean()
            # clamp the kld?
            kld = torch.clamp(kld, min=0.001)
            
            #
            z_mu_curr = z_mu_q
            z_logvar_curr  = z_logvar_q
        
        else:
            
            z_mu_curr = z_mu_p
            z_logvar_curr = z_logvar_p
            kld = 0
        
        
        ######## avoid the 'contiguous' problem
        #
        z_var_curr = (0.5 * z_logvar_curr).exp()
        #
        z_list = []
        for i in range(num_samples_forEachTraj):
            #
            curr_samples = torch.randn(past_h.shape[0], self.latent_dim).cuda()
            z_curr = z_mu_curr + curr_samples * z_var_curr
            #
            z_list.append(z_curr)
        
        
        return z_list, kld






######## 2021/05/17 suzx
# multihead attention! (maybe better then single-head attention)
class Transformer_Cross_Encoder(nn.Module):
    
    def __init__(self, feat_dim=272, modality_atten=True):
        super(Transformer_Cross_Encoder, self).__init__()
        
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
        self.trans_ego_with_bbox = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        #
        self.trans_bbox_with_pose = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        self.trans_pose_with_bbox = self.get_transformer_network(embed_dim=feat_dim, attn_dropout=self.attn_dropout)
        
        #
        self.modality_atten = modality_atten
    
    
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
        h_ego_with_pose, attn_weight_egoPose = self.trans_ego_with_pose(x_ego, x_pose, x_pose)    # Dimension (L, N, d_l)
        h_ego_with_bbox, attn_weight_egoBbox = self.trans_ego_with_bbox(x_ego, x_bbox, x_bbox)    # Dimension (L, N, d_l)
        #
        h_bbox_with_pose, attn_weight_bboxPose = self.trans_bbox_with_pose(x_bbox, x_pose, x_pose)    # Dimension (L, N, d_l)
        h_pose_with_bbox, attn_weight_poseBbox = self.trans_pose_with_bbox(x_pose, x_bbox, x_bbox)    # Dimension (L, N, d_l)
        
        #
        if self.modality_atten==False:
            h_cross = torch.cat([h_ego_with_bbox, h_ego_with_pose, h_bbox_with_pose, h_pose_with_bbox], dim=2)
            h_cross = h_cross.transpose(0, 1)
            return h_cross
        else:
            # (3.1) q2c
            crossAtten_q2c_stack = torch.cat([attn_weight_egoPose.unsqueeze(dim=2), attn_weight_egoBbox.unsqueeze(dim=2)], dim=2)
            modalityAtten_q2c = crossAtten_q2c_stack.max(dim=3)[0]
            modalityAtten_q2c = F.softmax(modalityAtten_q2c, dim=2)
            # (3.2) c2c
            crossAtten_c2c_stack = torch.cat([attn_weight_bboxPose.unsqueeze(dim=2), attn_weight_poseBbox.unsqueeze(dim=2)], dim=2)
            modalityAtten_c2c = crossAtten_c2c_stack.max(dim=3)[0]
            modalityAtten_c2c = F.softmax(modalityAtten_c2c, dim=2)
            #
            q2c_h = modalityAtten_q2c[:, :, 0].transpose(0, 1).unsqueeze(dim=2) * h_ego_with_pose + modalityAtten_q2c[:, :, 1].transpose(0, 1).unsqueeze(dim=2) * h_ego_with_bbox
            c2c_h = modalityAtten_c2c[:, :, 0].transpose(0, 1).unsqueeze(dim=2) * h_bbox_with_pose + modalityAtten_c2c[:, :, 1].transpose(0, 1).unsqueeze(dim=2) * h_pose_with_bbox
            h_cross = torch.cat([q2c_h, c2c_h], dim=2)
            h_cross = h_cross.transpose(0, 1)
            return h_cross










########
class Trans_Predictor_Multimodal(nn.Module):
     
    def __init__(self, cfg, dataset_name='', use_img_feat=True, use_attrib_feat=True, 
                 use_cross_atten=True, atten_manner='atten_then_concat'):
        super(Trans_Predictor_Multimodal, self).__init__()
         
        self.cfg = copy.deepcopy(cfg)
        self.pred_len = self.cfg.PRED_LEN
        self.param_scheduler = None    # this variable will be set in 'train.py'
        self.dataset_name = dataset_name
         
        #
        self.use_img_feat = use_img_feat
        self.use_attrib_feat = use_attrib_feat
        self.use_cross_atten = use_cross_atten
        self.use_modality_atten = self.cfg.USE_MODALITY_ATTEN
        self.atten_manner = atten_manner
         
        #
        dim_in_concat = 0
         
         
        ########
        if self.use_cross_atten:
            self.encoder = Transformer_Cross_Encoder(feat_dim=self.cfg.INPUT_EMBED_SIZE, modality_atten=self.use_modality_atten)
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
            if self.use_modality_atten:
                dim_in_concat = self.cfg.ENC_HIDDEN_SIZE * 2
            else:
                dim_in_concat = self.cfg.ENC_HIDDEN_SIZE * 4
        else:
            dim_in_concat = self.cfg.ENC_HIDDEN_SIZE
         
         
         
         
        ########
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
         
         
        #### latent net
        self.hidden_size = self.cfg.ENC_HIDDEN_SIZE
        if self.use_cross_atten:
            self.latent_net = Gaussian_Latent_Net(dim_pastFeat=dim_in_final, dim_embed=self.cfg.INPUT_EMBED_SIZE, dim_state=self.hidden_size, dim_in_ego=dim_in_ego, 
                                                  manner_construct_future_h='cross_atten', latent_dim=self.cfg.LATENT_DIM)
        else:
            self.latent_net = Gaussian_Latent_Net(dim_pastFeat=dim_in_final, dim_embed=self.cfg.INPUT_EMBED_SIZE, dim_state=self.hidden_size, dim_in_ego=dim_in_ego, 
                                                  manner_construct_future_h='concat', latent_dim=self.cfg.LATENT_DIM) 
         
         
        
        #
        self.intention_predictor = make_mlp(dim_list=[dim_in_final, int(dim_in_final/2), 1], activation='', batch_norm=False, bias=True, dropout=0)    # the loss i should use is: logistic loss (binary cross entropy)
        
        self.intention_predictor_v2 = nn.Sequential(nn.Linear(dim_in_final, int(dim_in_final/2)), 
                                                    nn.ReLU(), 
                                                    nn.Linear(int(dim_in_final/2), 64), 
                                                    nn.ReLU(), 
                                                    nn.Linear(64, self.cfg.INTENT_OUT_DIM)
                                                    )
        
        
         
        ####
        dim_in_final += self.cfg.LATENT_DIM    # it can be changed dependently.
        self.control_points_mlp_predictor = nn.Sequential(nn.Linear(dim_in_final, 512), 
                                                          nn.ReLU(), 
                                                          nn.Linear(512, 256), 
                                                          nn.ReLU(), 
                                                          nn.Linear(256, 12)
                                                          )
        
        
        
         
     
    def obtain_feat_crossmodal(self, img_inputs, obs_bbox=None, obs_ego=None):
        
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
            final_feats = self.encoder(feats_ego_encode, feats_img_encode, feats_bbox_encode)
        else:
            final_feats = self.encoder(embed_bbox_encode)
         
         
         
        return final_feats
     
     
     
    ########
    def obtain_feat_social_simple(self, obsLast_look, obsLast_body_ori, obsLast_bbox, obsLast_ego_act):
        social_attrib = torch.cat([obsLast_look, obsLast_body_ori, obsLast_bbox, obsLast_ego_act], dim=1)
        social_feats = self.social_layer_simple(social_attrib)
         
        return social_feats
     
     
     
    ########
    def traj_predict(self, obsLast_bbox, bbox_h_list):
        
        pred_traj_list = []
        #
        for bbox_h in bbox_h_list:
            
            #bbox_h = F.dropout(bbox_h, p=0.05, training=self.training)
            
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
            
            #
            pred_traj_list.append(pred_traj.unsqueeze(dim=2))
        #
        pred_trajs = torch.cat(pred_traj_list, dim=2)
        
        return pred_trajs
     
    
     
     
     
     
     
     
     
    def forward(self, img_inputs, look_inputs, body_ori_inputs, 
                obs_bbox=None, pred_bbox_gt=None, obs_ego=None, pred_ego_gt=None, 
                num_sample=20):
         
        #
        feats_atten = self.obtain_feat_crossmodal(img_inputs, obs_bbox, obs_ego)
         
        past_h_atten = feats_atten.max(dim=1)[0]    # if here i use 'max-pooling', the final-fusion-layer should activated by 'relu', not 'tanh'
        
        
        # social features
        if self.use_attrib_feat:
            if self.cfg.ATTRIB_FEAT_TYPE=='simple':
                social_feat = self.obtain_feat_social_simple(look_inputs, body_ori_inputs, obs_bbox[:, -1, :], obs_ego[:, -1])
            else:
                social_feat = None
                print('do not support complex social modeling.')
                assert(1==0)
            #
            feat_obs = torch.cat([past_h_atten, social_feat], dim=1)
        else:
            feat_obs = past_h_atten
        
        
        # latent net
        z_list, kld = self.latent_net(feat_obs, pred_bbox_gt, pred_ego_gt, num_samples_forEachTraj=num_sample)
        
        # decoding (multimodal prediction)
        noise_bbox_h_list = []
        for curr_z in z_list:
            curr_noise_bbox_h = torch.cat([feat_obs, curr_z], dim=1)
            noise_bbox_h_list.append(curr_noise_bbox_h)
        #
        pred_trajs = self.traj_predict(obs_bbox[:, -1, :], noise_bbox_h_list)
        
        
        #
        crossing_outputs = self.intention_predictor_v2(feat_obs)
         
        return crossing_outputs, pred_trajs, kld
     
     
     
     
    def predict(self, img_inputs, look_inputs, body_ori_inputs, 
                obs_bbox=None, pred_bbox_gt=None, obs_ego=None, pred_ego_gt=None, 
                num_sample=20):
         
        #
        feats_atten = self.obtain_feat_crossmodal(img_inputs, obs_bbox, obs_ego)
         
        past_h_atten = feats_atten.max(dim=1)[0]    # if here i use 'max-pooling', the final-fusion-layer should activated by 'relu', not 'tanh'
        
        
        # social features
        if self.use_attrib_feat:
            if self.cfg.ATTRIB_FEAT_TYPE=='simple':
                social_feat = self.obtain_feat_social_simple(look_inputs, body_ori_inputs, obs_bbox[:, -1, :], obs_ego[:, -1])
            else:
                social_feat = None
                print('do not support complex social modeling.')
                assert(1==0)
            #
            feat_obs = torch.cat([past_h_atten, social_feat], dim=1)
        else:
            feat_obs = past_h_atten
        
        
        # latent net
        z_list, kld = self.latent_net(feat_obs, pred_bbox_gt, pred_ego_gt, num_samples_forEachTraj=num_sample)
        
        # decoding (multimodal prediction)
        noise_bbox_h_list = []
        for curr_z in z_list:
            curr_noise_bbox_h = torch.cat([feat_obs, curr_z], dim=1)
            noise_bbox_h_list.append(curr_noise_bbox_h)
        #
        pred_trajs = self.traj_predict(obs_bbox[:, -1, :], noise_bbox_h_list)
        
        
        #
        prob_cross_softmax = F.softmax(self.intention_predictor_v2(feat_obs), dim=1)
        pred_cross = prob_cross_softmax.max(dim=1)[1]
        pred_cross = pred_cross.reshape([-1, 1])
         
        return pred_cross, pred_trajs







