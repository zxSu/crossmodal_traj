from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import pickle
import cv2
import copy
import json

from core.evaluate import accuracy
from core.evaluate import comp_deg_error, continous_comp_deg_error, draw_orientation, ori_numpy, draw_pred_orientation

logger = logging.getLogger(__name__)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
def train(config, train_loader, train_dataset, model, criterions, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_2d_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target, target_weight, degree, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # compute output
        plane_output, hoe_output = model(input)

        # change to cuda format
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        degree = degree.cuda(non_blocking=True)

        # compute loss
        if config.LOSS.USE_ONLY_HOE:
            loss_hoe = criterions['hoe_loss'](hoe_output, degree)
            loss_2d = loss_hoe
            loss = loss_hoe
        else:
            loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
            loss_hoe = criterions['hoe_loss'](hoe_output , degree)

            loss = loss_2d + 0.1*loss_hoe

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss_2d_log.update(loss_2d.item(), input.size(0))
        loss_hoe_log.update(loss_hoe.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        if config.DATASET.DATASET == 'tud_dataset':
            avg_degree_error, _, _, _ , _, _, _, _= continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                               meta['val_dgree'].numpy())
        else:
            avg_degree_error, _, _, _ , _, _, _, _= comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                   degree.detach().cpu().numpy())
            _, avg_acc, cnt, pred = accuracy(plane_output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

        degree_error.update(avg_degree_error, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss_hkd {loss_hkd.val:.5f} ({loss_hkd.avg:.5f})\t' \
                  'Loss_hoe {loss_hoe.val:.5f} ({loss_hoe.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Degree_error {Degree_error.val:.3f} ({Degree_error.avg:.3f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss_hkd=loss_2d_log, loss_hoe=loss_hoe_log, loss=losses,
                Degree_error=degree_error, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_hkd_loss', loss_2d_log.val, global_steps)
            writer.add_scalar('train_hoe_loss', loss_hoe_log.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer.add_scalar('degree_error', degree_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


# this is validate part
def validate(config, val_loader, val_dataset, model, criterions,  output_dir,
             tb_log_dir, writer_dict=None, draw_pic=False, save_pickle=False):
    batch_time = AverageMeter()
    loss_hkd_log = AverageMeter()
    loss_hoe_log = AverageMeter()
    losses = AverageMeter()
    degree_error = AverageMeter()
    acc = AverageMeter()
    Excellent = 0
    Mid_good = 0
    Poor_good = 0
    Poor_225 = 0
    Poor_45 = 0
    Total = 0

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    ori_list = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, degree,  meta) in enumerate(val_loader):
            # compute output
            plane_output, hoe_output = model(input)

            # change to cuda format
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            degree = degree.cuda(non_blocking=True)

            # compute loss
            if config.LOSS.USE_ONLY_HOE:
                loss_hoe = criterions['hoe_loss'](hoe_output, degree)
                loss_2d = loss_hoe
                loss = loss_hoe
            else:
                loss_2d = criterions['2d_pose_loss'](plane_output, target, target_weight)
                loss_hoe = criterions['hoe_loss'](hoe_output, degree)

                loss = loss_2d + 0.1 * loss_hoe

            num_images = input.size(0)
            # measure accuracy and record loss
            loss_hkd_log.update(loss_2d.item(), num_images)
            loss_hoe_log.update(loss_hoe.item(), num_images)
            losses.update(loss.item(), num_images)

            if 'tud' in config.DATASET.VAL_ROOT:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45, gt_ori, pred_ori  = continous_comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                                                   meta['val_dgree'].numpy())
            else:
                avg_degree_error, excellent, mid, poor_225, poor, poor_45,gt_ori, pred_ori  = comp_deg_error(hoe_output.detach().cpu().numpy(),
                                                           degree.detach().cpu().numpy())
                _, avg_acc, cnt, pred = accuracy(plane_output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)

            if draw_pic:
                ori_path = os.path.join(output_dir, 'orientation_img')
                if not os.path.exists(ori_path):
                    os.makedirs(ori_path)
                img_np = input.numpy()
                draw_orientation(img_np, gt_ori, pred_ori , ori_path, alis=str(i))

            if save_pickle:
                tamp_list = ori_numpy(gt_ori, pred_ori)
                ori_list = ori_list + tamp_list

            degree_error.update(avg_degree_error, num_images)

            Total += num_images
            Excellent += excellent
            Mid_good += mid
            Poor_good += poor
            Poor_45 += poor_45
            Poor_225 += poor_225

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss_hkd {loss_hkd.val:.5f} ({loss_hkd.avg:.5f})\t' \
                      'Loss_hoe {loss_hoe.val:.5f} ({loss_hoe.avg:.5f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Degree_error {Degree_error.val:.3f} ({Degree_error.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss_hkd=loss_hkd_log, loss_hoe=loss_hoe_log, loss=losses, Degree_error = degree_error, acc=acc)
                logger.info(msg)

        if save_pickle:
            save_obj(ori_list, 'ori_list')
        excel_rate = Excellent / Total
        mid_rate = Mid_good / Total
        poor_rate = Poor_good / Total
        poor_225_rate = Poor_225 / Total
        poor_45_rate = Poor_45 / Total
        name_values = {'Degree_error': degree_error.avg, '5_Excel_rate': excel_rate, '15_Mid_rate': mid_rate, '225_rate': poor_225_rate, '30_Poor_rate': poor_rate, '45_poor_rate': poor_45_rate}
        _print_name_value(name_values, config.MODEL.NAME)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hkd_loss',
                loss_hkd_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hoe_loss',
                loss_hoe_log.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            writer.add_scalar(
                'degree_error_val',
                degree_error.avg,
                global_steps
            )
            writer.add_scalar(
                'excel_rate',
                excel_rate,
                global_steps
            )
            writer.add_scalar(
                'mid_rate',
                mid_rate,
                global_steps
            )
            writer.add_scalar(
                'poor_rate',
                poor_rate,
                global_steps
            )

            writer_dict['valid_global_steps'] = global_steps + 1

        perf_indicator = degree_error.avg
    return perf_indicator





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





def my_demo(cfg, transform, model, output_dir, save_json=True):
    
    #
    img_width = cfg.MODEL.IMAGE_SIZE[0]
    img_height = cfg.MODEL.IMAGE_SIZE[1]
    aspect_ratio = img_width * 1.0 / img_height
    pixel_std = 200
    
    input_root = '/home/suzx/new_disks/eclipse_ws2/PIEpredict/PIE_dataset/img_feat_cache/set01/video_0002'
    
    # switch to evaluate mode
    model.eval()
    
    ori_dict = {}
    with torch.no_grad():
        
        # (1) get all files under the 'input root'
        all_filepaths = [ os.path.join(input_root, f) for f in os.listdir(input_root) ]
        
        # (2) load image
        for curr_npz_path in all_filepaths:
            
#             with open(curr_npz_path, 'rb') as fid:
#                 try:
#                     curr_input_np = pickle.load(fid)
#                 except:
#                     curr_input_np = pickle.load(fid, encoding='bytes')
            
            npz_data = np.load(curr_npz_path)
            curr_input_np = npz_data['img_feat']
            
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
            _, hoe_output = model(curr_input_tensor)
            
            # output to pred_degree
            index_degree = hoe_output.argmax(axis = 1)
            pred_ori = index_degree * 5
            pred_ori = pred_ori.detach().cpu().numpy()
            
            #
            set_id = curr_npz_path.split('/')[-3]
            vid_id = curr_npz_path.split('/')[-2]
            img_name = curr_npz_path.split('/')[-1].split('.')[0]
            
            # visualize orientation
            curr_ori_folder = os.path.join(output_dir, set_id, vid_id)
            if not os.path.exists(curr_ori_folder):
                os.makedirs(curr_ori_folder)
            draw_pred_orientation(curr_input_original, pred_ori , curr_ori_folder, img_name)
            
            #
            if save_json:
                ori_dict[img_name] = str(pred_ori[0])
        
        # write all predicted results to a '.json' file
        curr_ori_txt_path = os.path.join(output_dir, set_id, vid_id+'_ori.json')
        with open(curr_ori_txt_path, 'w') as f:
            json.dump(ori_dict, f)








def my_demo_v2(cfg, img_np_list, img_path_list, transform, model, output_dir, save_json=True):
    
    #
    img_width = cfg.MODEL.IMAGE_SIZE[0]
    img_height = cfg.MODEL.IMAGE_SIZE[1]
    aspect_ratio = img_width * 1.0 / img_height
    pixel_std = 200
    
    # switch to evaluate mode
    model.eval()
    
    ori_dict = {}
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
            _, hoe_output = model(curr_input_tensor)
            
            # output to pred_degree
            index_degree = hoe_output.argmax(axis = 1)
            pred_ori = index_degree * 5
            pred_ori = pred_ori.detach().cpu().numpy()
            
            #
            set_id = curr_path.split('/')[-3]
            vid_id = curr_path.split('/')[-2]
            img_name = curr_path.split('/')[-1].split('.')[0]
            
            # visualize orientation
            curr_ori_folder = os.path.join(output_dir, set_id, vid_id)
            if not os.path.exists(curr_ori_folder):
                os.makedirs(curr_ori_folder)
            draw_pred_orientation(curr_input_original, pred_ori , curr_ori_folder, img_name)
            
            #
            if save_json:
                ori_dict[img_name] = str(pred_ori[0])
                
    # need 'seq_id'
        
#         # write all predicted results to a '.json' file
#         curr_ori_txt_path = os.path.join(output_dir, set_id, vid_id+'_ori.json')
#         with open(curr_ori_txt_path, 'w') as f:
#             json.dump(ori_dict, f)














# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
