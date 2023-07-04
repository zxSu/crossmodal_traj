import numpy as np
import pickle
import os
import PIL
import copy

import torch.nn as nn
import torch
from torchvision.ops import roi_align

from datasets.MEBOW_utils import mebow_evaluate

from torchvision import transforms

import cv2



def bbox_sanity_check(img_height, img_width, bbox):
    '''
    This is to confirm that the bounding boxes are within image boundaries.
    If this is not the case, modifications is applied.
    This is to deal with inconsistencies in the annotation tools
    '''
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_height:
        bbox[3] = img_height - 1
    return bbox


    
    
    
def jitter_bbox(img_height, img_width, bbox, mode, ratio):
    '''
    This method jitters the position or dimentions of the bounding box.
    mode: 'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
    ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
    '''
    assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
            'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio  = abs(ratio)
    else:
        jitter_ratio  = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample()*jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge','random_enlarge']:
            b[0] = b[0] - width_change //2
            b[1] = b[1] - height_change //2
        else:
            b[0] = b[0] + width_change //2
            b[1] = b[1] + height_change //2

        b[2] = b[2] + width_change //2
        b[3] = b[3] + height_change //2

        # Checks to make sure the bbox is not exiting the image boundaries
        b =  bbox_sanity_check(img_height, img_width, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes



def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2
    # bbox[1] = str(float(bbox[1]) - width_change/2)
    # bbox[3] = str(float(bbox[3]) + width_change)
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0]-bbox[2] + img_width
        bbox[2] = img_width
    return bbox




def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image










def generate_img_feats_v4(img_sequences, 
                             bbox_sequences, 
                             ped_ids, 
                             ori_save_root, 
                             feat_save_root, 
                             data_transform, 
                             mebow_estimator, 
                             mebow_cfg):
    
    ####
    sequences_ori = []
    sequences_feat = []
    i = -1
    for seq, pid in zip(img_sequences, ped_ids):
        i += 1
        
        imp_end = seq[-1]
        set_id = imp_end.split('/')[-3]
        vid_id = imp_end.split('/')[-2]
        name_end = imp_end.split('/')[-1].split('.')[0]
        #
        ori_save_folder = os.path.join(ori_save_root, set_id, vid_id)
        ori_save_path = os.path.join(ori_save_folder, pid[0][0]+'_'+name_end+'.npz')
        #
        feat_save_folder = os.path.join(feat_save_root, set_id, vid_id)
        feat_save_path = os.path.join(feat_save_folder, pid[0][0]+'_'+name_end+'.npz')
        
        if os.path.exists(ori_save_path) and os.path.exists(feat_save_path):
            #
            ori_results = np.load(ori_save_path)['ori_results']
            pose_results = np.load(feat_save_path)['pose_results']
            print('estimated human body orientation are existed in: '+ori_save_path)
            print('estimated pose heatmaps are existed in: '+feat_save_path)
        
        else:
            #
            if not os.path.exists(ori_save_folder):
                os.makedirs(ori_save_folder)
            if not os.path.exists(feat_save_folder):
                os.makedirs(feat_save_folder)
            #
            img_np_list = []
            for imp, b, p in zip(seq, bbox_sequences[i], pid):
                #img_data = load_img(imp)
                img_data = PIL.Image.open(imp)
                bbox_ped = b
                
                #bbox_w = abs(b[2] - b[0])
                #bbox_h = abs(b[3] - b[1])
                
                #
                cropped_image = img_data.crop(bbox_ped)                     
                img_np = np.array(cropped_image)
                #
                img_np_list.append(img_np)   
            #
            ori_result_list, pose_result_list = mebow_evaluate(mebow_cfg, img_np_list, seq, data_transform, mebow_estimator, ori_save_folder)
            # save ori
            ori_results = np.array(ori_result_list)
            np.savez(ori_save_path, ori_results=ori_results, pid=pid[0])
            print('estimate human body orientation and save to: '+ori_save_path)
            # save pose
            pose_results = np.array(pose_result_list)
            np.savez(feat_save_path, pose_results=pose_results, pid=pid[0])
            print('estimate human pose heatmaps and save to: '+feat_save_path)
                
        sequences_ori.append(ori_results)
        sequences_feat.append(pose_results)
    #
    return sequences_feat, sequences_ori
































