import os
import numpy as np
import torch
from bitrap.modeling.gmm4d import GMM4D
import PIL

from bitrap.utils.box_utils import cxcywh_to_x1y1x2y2

import matplotlib.pyplot as plt
from PIL import Image
import cv2


def create_folder(path):
    # check whether the folder is exist or not
    if os.path.exists(path):
        #print('folder exist')
        pass
    else:
        #print('folder not exist, so i will create it')
        os.makedirs(path)



def print_info(epoch, model, optimizer, loss_dict, logger):
    loss_dict['kld_weight'] = model.param_scheduler.kld_weight.item()
    loss_dict['z_logit_clip'] = model.param_scheduler.z_logit_clip.item()

    info = "Epoch:{},\t lr:{:6},\t loss_goal:{:.4f},\t loss_traj:{:.4f},\t loss_kld:{:.4f},\t \
            kld_w:{:.4f},\t z_clip:{:.4f} ".format( 
            epoch, optimizer.param_groups[0]['lr'], loss_dict['loss_goal'], loss_dict['loss_traj'], 
            loss_dict['loss_kld'], loss_dict['kld_weight'], loss_dict['z_logit_clip']) 
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)

def viz_results(viz, 
                X_global, 
                y_global, 
                pred_traj, 
                img_path, 
                dist_goal, 
                dist_traj,
                bbox_type='cxcywh',
                normalized=True,
                logger=None, 
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    id_to_show = np.random.randint(pred_traj.shape[0])

    # 1. initialize visualizer
    viz.initialize(img_path[id_to_show])

    # 2. visualize point trajectory or box trajectory
    if y_global.shape[-1] == 2:
        viz.visualize(pred_traj[id_to_show], color=(0, 1, 0), label='pred future', viz_type='point')
        viz.visualize(X_global[id_to_show], color=(0, 0, 1), label='past', viz_type='point')
        viz.visualize(y_global[id_to_show], color=(1, 0, 0), label='gt future', viz_type='point')
    elif y_global.shape[-1] == 4:
        T = X_global.shape[1]
        viz.visualize(pred_traj[id_to_show], color=(0, 255., 0), label='pred future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])
        viz.visualize(X_global[id_to_show], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(y_global[id_to_show], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])        

    # 3. optinaly visualize GMM distribution
    if hasattr(dist_goal, 'mus') and viz.mode == 'plot':
        dist = {'mus':dist_goal.mus.numpy(), 'log_pis':dist_goal.log_pis.numpy(), 'cov': dist_goal.cov.numpy()}
        viz.visualize(dist, id_to_show=id_to_show, viz_type='distribution')
    
    # 4. get image. 
    if y_global.shape[-1] == 2:
        viz_img = viz.plot_to_image(clear=True)
    else:
        viz_img = viz.img

    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)




####
def my_viz_results(viz, 
                X_global, 
                y_global, 
                pred_traj, 
                img_path, 
                result_save_root, 
                curr_iter, 
                test_batch_size, 
                bbox_type='cxcywh',
                normalized=True, 
                pred_crossing=None, 
                gt_crossing=None):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    
    create_folder(result_save_root)
    
    num_traj = pred_traj.shape[0]
    
    for i in range(num_traj):
        
        # 1. initialize visualizer
        viz.initialize(img_path[i])
    
        # 2. visualize point trajectory or box trajectory
        if y_global.shape[-1] == 2:
            viz.visualize(pred_traj[i], color=(0, 1, 0), label='pred future', viz_type='point')
            viz.visualize(X_global[i], color=(0, 0, 1), label='past', viz_type='point')
            viz.visualize(y_global[i], color=(1, 0, 0), label='gt future', viz_type='point')
        elif y_global.shape[-1] == 4:
            T = X_global.shape[1]
#             viz.visualize(pred_traj[i], color=(0, 255., 0), label='pred future', viz_type='bbox', 
#                           normalized=normalized, bbox_type=bbox_type, viz_time_step=[0])
            viz.visualize(pred_traj[i], color=(0, 255., 0), label='pred future', viz_type='bbox', 
                          normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1], crossing_flag=pred_crossing[i, 0])    #
            viz.visualize(X_global[i], color=(0, 0, 255.), label='past', viz_type='bbox', 
                          normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
            viz.visualize(y_global[i], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                          normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1], crossing_flag=gt_crossing[i, 0])    #
            
        
        #### my adding
        # get 'set_id' and 'video_id'
        set_id = img_path[i].split('/')[-3]
        vid_id = img_path[i].split('/')[-2]
        img_name = img_path[i].split('/')[-1].split('.')[0]
        curr_result_save_root = os.path.join(result_save_root, 'img', set_id, vid_id)
        curr_result_save_path = os.path.join(curr_result_save_root, str(curr_iter*test_batch_size+i)+'_'+img_name+'.jpg')
        create_folder(curr_result_save_root)
        viz.write_result_img(curr_result_save_path)
        print('the trajectory '+str(i)+', writing the visualization result to file: '+curr_result_save_path)
        
        

####
def my_viz_results_v2(viz, 
                      y_global, 
                      img_path, 
                      curr_iter, 
                      test_batch_size, 
                      kde_instance_list, 
                      idxInBatch_list, 
                      result_save_root, 
                      img_shape_new=None):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    
    #result_save_root = '/home/suzx/new_disks/eclipse_ws2/PIEpredict/result_visualize_v2'
    create_folder(result_save_root)
    
    num_traj = len(kde_instance_list)
    
    for i in range(num_traj):
        
        # get 'set_id' and 'video_id'
        set_id = img_path[i].split('/')[-3]
        vid_id = img_path[i].split('/')[-2]
        img_name = img_path[i].split('/')[-1].split('.')[0]
        idx_in_batch = idxInBatch_list[i]
        result_filename = str(curr_iter*test_batch_size+idx_in_batch)+'_'+img_name
        
        # 1. initialize visualizer
        viz.initialize_v2(img_path[i], img_shape_new)
        
        # 2. current kde instance
        curr_kde_instance = kde_instance_list[i]
        
        # 3. visualize point trajectory or box trajectory
        viz.visualize(y_global[i], color=(255., 0, 0), label='gt future', viz_type='bbox_only', 
                      normalized=False, bbox_type='cxcywh', viz_time_step=[-1])    #
        viz.visualize(None, viz_type='point_goalMap', kde_instance=curr_kde_instance)
        
        # save
        curr_result_save_root = os.path.join(result_save_root, 'img', set_id, vid_id)
        curr_result_save_path = os.path.join(curr_result_save_root, result_filename+'.jpg')
        create_folder(curr_result_save_root)
        viz.write_result_img(curr_result_save_path)
        print('the trajectory '+str(i)+', writing the visualization result to file: '+curr_result_save_path)






#### 2021/09/08
def my_viz_results_v3(viz, 
                X_global, 
                y_global, 
                pred_traj_bitrap, 
                pred_traj_ours, 
                img_path, 
                result_save_root, 
                bbox_type='cxcywh',
                normalized=True):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    
    create_folder(result_save_root)
    
    num_traj = X_global.shape[0]
    
    for i in range(num_traj):
        
        # 1. initialize visualizer
        viz.initialize(img_path[i])
        
        # 2. box trajectory
        T = X_global.shape[1]
        viz.visualize(X_global[i], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, thickness=2, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(pred_traj_bitrap[i], color=(0, 255., 0), label='bitrap', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, thickness=5, viz_time_step=[-1])    #
        viz.visualize(y_global[i], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, thickness=5, viz_time_step=[-1])    #
        viz.visualize(pred_traj_ours[i], color=(255, 255, 0), label='ours', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, thickness=5, viz_time_step=[-1])    #
            
        
        #### my adding
        # get 'set_id' and 'video_id'
        set_id = img_path[i].split('/')[-3]
        vid_id = img_path[i].split('/')[-2]
        img_name = img_path[i].split('/')[-1].split('.')[0]
        curr_result_save_root = os.path.join(result_save_root, 'img', set_id, vid_id)
        curr_result_save_path = os.path.join(curr_result_save_root, str(i)+'_'+img_name+'.jpg')
        create_folder(curr_result_save_root)
        viz.write_result_img(curr_result_save_path)
        print('the trajectory '+str(i)+', writing the visualization result to file: '+curr_result_save_path)




#### 2021/09/14
def my_viz_results_v4(viz, 
                      cfg, 
                      iters, 
                      X_global, 
                      img_path, 
                      obsLast_ped_look, 
                      obsLast_body_ori, 
                      obs_ego, 
                      result_save_root, 
                      bbox_type='cxcywh', 
                      normalized=True, 
                      select_example=None, 
                      attenWeights_tuple=None):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    
    create_folder(result_save_root)
    
    num_traj = X_global.shape[0]
    
    for i in range(num_traj):
        
        
        # 0. draw the example that user define
        vid_id = img_path[i].split('/')[-2]
        img_name = img_path[i].split('/')[-1].split('.')[0]
        result_filename = str((iters-1)*cfg.TEST.BATCH_SIZE+i)+'_'+img_name
        # whether we need to draw distribution for this trajectory
        if select_example is not None:
            save_flag = False
            for curr_example_info in select_example:
                if (vid_id==curr_example_info[0]) and (result_filename==curr_example_info[1]):
                    save_flag = True
                    break
            #
            if save_flag==False:
                # this example is not we wanted
                continue
        
        # 1. initialize visualizer
        viz.initialize(img_path[i])
        
        # 2. box trajectory
        T = X_global.shape[1]
        viz.visualize(X_global[i], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        
        
        # create folder
        set_id = img_path[i].split('/')[-3]
        curr_result_save_root = os.path.join(result_save_root, 'img', set_id, vid_id)
        curr_img_folder = os.path.join(curr_result_save_root, result_filename)
        create_folder(curr_result_save_root)
        create_folder(curr_img_folder)
        
        
        frameId_currTrajEnd = int(img_name)
        rootElem_currTraj_list = img_path[i].split('/')[1:-1]
        root_currTraj = '/'
        for curr_root_elem in rootElem_currTraj_list:
            root_currTraj = os.path.join(root_currTraj, curr_root_elem)
        
        for t in range(15):
            curr_frameId = frameId_currTrajEnd - 14 + t
            curr_imgName = str(curr_frameId).zfill(5) + '.png'
            curr_imgPath = os.path.join(root_currTraj, curr_imgName)
            # crop now
            img_data = PIL.Image.open(curr_imgPath)
            curr_bbox = X_global[i][t, :]
            # cxcywh -> x1y1x2y2
            curr_bbox = cxcywh_to_x1y1x2y2(curr_bbox)
            cropped_image = img_data.crop(curr_bbox)
            # save to folder
            curr_crop_img_path = os.path.join(curr_img_folder, curr_imgName)
            cropped_image.save(curr_crop_img_path)
            
        
#         # crop now
#         img_data = PIL.Image.open(img_path[i])
#         cropped_image = img_data.crop(curr_bbox)
#         # save to folder
#         curr_crop_img_path = os.path.join(curr_img_folder, '14.jpg')
#         cropped_image.save(curr_crop_img_path)
#         
#         ####
#         curr_bbox = X_global[i][8, :]
#         # cxcywh -> x1y1x2y2
#         curr_bbox = cxcywh_to_x1y1x2y2(curr_bbox)
#         # crop now
#         img_data = PIL.Image.open(img_path[i-1])
#         cropped_image = img_data.crop(curr_bbox)
#         # save to folder
#         curr_crop_img_path = os.path.join(curr_img_folder, '8.jpg')
#         cropped_image.save(curr_crop_img_path)
#         
#         ####
#         curr_bbox = X_global[i][1, :]
#         # cxcywh -> x1y1x2y2
#         curr_bbox = cxcywh_to_x1y1x2y2(curr_bbox)
#         # crop now
#         img_data = PIL.Image.open(img_path[i-2])
#         cropped_image = img_data.crop(curr_bbox)
#         # save to folder
#         curr_crop_img_path = os.path.join(curr_img_folder, '1.jpg')
#         cropped_image.save(curr_crop_img_path)
            
        
        #
        curr_result_save_path = os.path.join(curr_result_save_root, result_filename+'.jpg')
        viz.write_result_img(curr_result_save_path)
        print('the trajectory '+str(i)+', writing the visualization result to file: '+curr_result_save_path)
        
        #
        fig, ax = plt.subplots(figsize=(8, 8))
        attenWeights_1_np = attenWeights_tuple[0][i, :].cpu().numpy()
        attenWeights_1_np = np.round(attenWeights_1_np, 2)
        im = ax.imshow(attenWeights_1_np.T)
        #
        dim1 = attenWeights_1_np.shape[0]
        dim2 = attenWeights_1_np.shape[1]
        x_label_list = obs_ego[i].tolist()
        #
        ax.set_xticks(np.arange(dim1))
        #ax.set_yticks(np.arange(dim2))
        ax.set_xticklabels(x_label_list)
        #ax.set_yticklabels()
        #
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')\
        #
        for m in range(dim1):
            for n in range(dim2):
                text = ax.text(m, n, attenWeights_1_np[m, n], ha='center', va='center', color='w')
        #
        ax.set_title('component weights in branch-1')
        fig.tight_layout()
        #plt.show()
        
        # save to folder
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        fig_image = Image.frombytes("RGBA", (w, h), buf.tostring())
        fig_np = np.asarray(fig_image)[:, :, (2,1,0)]
        #
        atten_result_save_root = result_save_root+'_2'
        create_folder(atten_result_save_root)
        atten_result_save_path = os.path.join(atten_result_save_root, set_id+'_'+vid_id+'_'+result_filename+'_atten.jpg')
        cv2.imwrite(atten_result_save_path, fig_np)
        
        plt.close()








def post_process(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    if pred_goal is not None:
        pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if hasattr(dist_traj, 'mus'):
        dist_traj.to('cpu')
        dist_traj.squeeze(1)
    if hasattr(dist_goal, 'mus'):
        dist_goal.to('cpu')
        dist_goal.squeeze(1)
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
        _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        if cfg.DATASET.NORMALIZE == 'zero-one':
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
            if pred_goal is not None:
                pred_goal = (pred_goal + 1) * (_max - _min)/2 + _min
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.DATASET.NORMALIZE == 'none':
            pass
        else:
            raise ValueError()

        # NOTE: June 19, convert distribution from cxcywh to image resolution x1y1x2y2
        if hasattr(dist_traj, 'mus') and cfg.DATASET.NORMALIZE != 'none':
        
            _min = torch.FloatTensor(cfg.DATASET.MIN_BBOX)[None, None, :].repeat(batch_size, T, 1) # B, T, dim
            _max = torch.FloatTensor(cfg.DATASET.MAX_BBOX)[None, None, :].repeat(batch_size, T, 1)
            zeros = torch.zeros_like(_min[..., 0])
            
            if cfg.DATASET.NORMALIZE == 'zero-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0], zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1], zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2], zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]], dim=-1),
                                ], dim=-2)
                b = torch.tensor(_min)
            elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0]/2, zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1]/2, zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2]/2, zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]/2], dim=-1),
                                ], dim=-2)
                b = torch.stack([(_max+_min)[..., 0]/2, (_max+_min)[..., 1]/2, (_max+_min)[..., 2]/2, (_max+_min)[..., 3]/2],dim=-1)
            try:
                traj_mus = torch.matmul(A.unsqueeze(2), dist_traj.mus.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(2)
                traj_cov = torch.matmul(A.unsqueeze(2), dist_traj.cov).matmul(A.unsqueeze(2).transpose(-1,-2))
                goal_mus = torch.matmul(A[:, 0:1, :], dist_goal.mus.unsqueeze(-1)).squeeze(-1) + b[:, 0:1, :]
                goal_cov = torch.matmul(A[:, 0:1, :], dist_goal.cov).matmul(A[:,0:1,:].transpose(-1,-2))
            except:
                raise ValueError()

            dist_traj = GMM4D.from_log_pis_mus_cov_mats(dist_traj.input_log_pis, traj_mus, traj_cov)
            dist_goal = GMM4D.from_log_pis_mus_cov_mats(dist_goal.input_log_pis, goal_mus, goal_cov)
    return X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal






def post_process_v2(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    if pred_goal is not None:
        pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if hasattr(dist_traj, 'mus'):
        dist_traj.to('cpu')
        dist_traj.squeeze(1)
    if hasattr(dist_goal, 'mus'):
        dist_goal.to('cpu')
        dist_goal.squeeze(1)
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
        _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        if cfg.DATASET.NORMALIZE == 'zero-one':
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
            if pred_goal is not None:
                pred_goal = (pred_goal + 1) * (_max - _min)/2 + _min
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.DATASET.NORMALIZE == 'none':
            pass
        else:
            raise ValueError()
    
    return X_global, y_global, pred_goal, pred_traj






def post_process_v3(cfg, X_global, y_global, pred_traj, bound_min=None, bound_max=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if dim == 4:
        # BBOX: denormalize and change the mode
        
        if (bound_min is not None) and (bound_max is not None):
            _min = np.array(bound_min)[None, None, :] # B, T, dim
            _max = np.array(bound_max)[None, None, :]
        else:
            _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
            _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        if cfg.DATASET.NORMALIZE == 'zero-one':
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.DATASET.NORMALIZE == 'plus-minus-one':
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.DATASET.NORMALIZE == 'none':
            pass
        else:
            raise ValueError()
    else:
        print('implement it later.')
        assert(1==0)
    
    return X_global, y_global, pred_traj







