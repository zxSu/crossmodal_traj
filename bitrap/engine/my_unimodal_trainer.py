import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import math

from bitrap.utils.visualization import Visualizer
from bitrap.utils.box_utils import cxcywh_to_x1y1x2y2
from bitrap.utils.dataset_utils import restore
from bitrap.modeling.gmm2d import GMM2D
from bitrap.modeling.gmm4d import GMM4D
from .evaluate import evaluate_unimodal, evaluate_multimodal
from .utils import print_info, viz_results, post_process, post_process_v3, my_viz_results, my_viz_results_v2, my_viz_results_v4

from tqdm import tqdm
import pickle as pkl

from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss, MSELoss
from torch.nn import CrossEntropyLoss    # i do not know whether 'torch.nn.CrossEntropyLoss' is different from 'torch.nn.modules.loss.CrossEntropyLoss'
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


from bitrap.modeling.my_net_utils import rmse_loss, exp_rmse_loss


ci_loss_fn = BCELoss()
ci_loss_fn_2 = BCEWithLogitsLoss()    # binary-class classification: crossing, not-crossing
ci_loss_fn_3 = CrossEntropyLoss()    # multi-class classfication: crossing, not-crossing, unknown
traj_loss_fn = MSELoss()


######## there are two datasets available which are 'PIE' and 'JAAD'.
def unimodal_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None, 
             visdom_viz=None, t=None, 
             consider_crossIntent=True, dataset_name='PIE'):
    model.train()
    max_iters = len(dataloader)
    
    train_step_global = t
    
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            #
            obs_bbox = batch['input_x'].to(device)
            pred_bbox_gt = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            
            #
            #obs_img_feats = batch['obs_img_feat'].to(device)
            obs_img_feats = batch['obs_pose_feat'].to(device)
            intention_binary = batch['intention_binary'].to(device)
            obsLast_ped_look = batch['obs_ped_look'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = batch['obs_body_ori'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = obsLast_body_ori * math.pi / 180.0
            
            
            ########
            if dataset_name=='PIE':
                #
                obs_sensor_speed = batch['obs_obd_speed'].to(device)
                obs_sensor_heading_angle = batch['obs_heading_angle'].to(device)
                obs_ego = torch.cat([obs_sensor_speed, obs_sensor_heading_angle], dim=-1)
            else:
                #
                obs_ego = batch['obs_ego_act'].to(device)
            
            
            # inference
            crossing_outputs, pred_traj = model(obs_img_feats, obsLast_ped_look, obsLast_body_ori, 
                                                obs_bbox=obs_bbox, obs_ego=obs_ego)
            
            #
            #crossing_loss = ci_loss_fn_2(crossing_outputs, intention_binary)
            crossing_loss = ci_loss_fn_3(crossing_outputs, intention_binary.squeeze(dim=1).long())
            #traj_loss = traj_loss_fn(pred_traj, pred_bbox_gt)
            #traj_loss = rmse_loss(pred_traj, pred_bbox_gt)
            traj_loss = exp_rmse_loss(pred_traj, pred_bbox_gt)
            
            #
            if consider_crossIntent:
                final_loss = crossing_loss * 2 + traj_loss
            else:
                final_loss = traj_loss
            
            model.param_scheduler.step()    # parameters scheduler, such as 'kld_weight'.
            
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            final_loss.backward()
            
            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            
            if cfg.SOLVER.scheduler == 'exp':
                lr_scheduler.step()
            
            # visdom visualization
            if (visdom_viz is not None) and (t is not None):
                visdom_viz.line([crossing_loss.item()], [train_step_global], win='ci_train', update='append')
                visdom_viz.line([traj_loss.item()], [train_step_global], win='traj_train', update='append')
                #
                train_step_global += 1
    
    #
    return train_step_global
            


def unimodal_val(cfg, epoch, model, dataloader, device, logger=None, 
           visdom_viz=None, t=None, 
           consider_crossIntent=False, dataset_name='PIE'):
    model.eval()
    loss_val = 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            #
            obs_bbox = batch['input_x'].to(device)
            pred_bbox_gt = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            
            #
            #obs_img_feats = batch['obs_img_feat'].to(device)
            obs_img_feats = batch['obs_pose_feat'].to(device)
            intention_binary = batch['intention_binary'].to(device)
            obsLast_ped_look = batch['obs_ped_look'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = batch['obs_body_ori'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = obsLast_body_ori * math.pi / 180.0
            
            ########
            if dataset_name=='PIE':
                #
                obs_sensor_speed = batch['obs_obd_speed'].to(device)
                obs_sensor_heading_angle = batch['obs_heading_angle'].to(device)
                obs_ego = torch.cat([obs_sensor_speed, obs_sensor_heading_angle], dim=-1)
            else:
                #
                obs_ego = batch['obs_ego_act'].to(device)
            
            
            # inference
            crossing_outputs, pred_traj = model(obs_img_feats, obsLast_ped_look, obsLast_body_ori, 
                                                obs_bbox=obs_bbox, obs_ego=obs_ego)
            
            #
            #crossing_loss = ci_loss_fn_2(crossing_outputs, intention_binary)
            crossing_loss = ci_loss_fn_3(crossing_outputs, intention_binary.squeeze(dim=1).long())
            #traj_loss = traj_loss_fn(pred_traj, pred_bbox_gt)
            #traj_loss = rmse_loss(pred_traj, pred_bbox_gt)
            traj_loss = exp_rmse_loss(pred_traj, pred_bbox_gt)
            
            if consider_crossIntent:
                loss_val += (traj_loss + crossing_loss * 2).item()
            else:
                loss_val += traj_loss.item()
            
    
    loss_val /= (iters + 1)
    
    info = "loss_val:{:.4f}".format(loss_val)
    print(info)
    
    # visdom visualization
    if (visdom_viz is not None) and (t is not None):
        visdom_viz.line([loss_val], [t], win='ci_val', update='append')
    
    return loss_val



def unimodal_inference(cfg, epoch, model, dataloader, device, logger=None, test_mode=False, 
                 visdom_viz=None, t=None, dataset_name='PIE', traj_viz=False, eval_kde_nll=False):
    model.eval()
    #
    pred_crossing_list = []
    gt_crossing_list = []
    #
    all_img_paths = []
    all_X_globals = []
    all_pred_trajs = []
    all_gt_trajs = []
    all_timesteps = []
    c_ade_inference = 0.0
    c_fde_inference = 0.0
    ade_5_inference = 0.0
    ade_10_inference = 0.0
    ade_15_inference = 0.0
    avgTime_iter = 0.0
    total_iter = 0.0
    #
    viz = Visualizer(mode='image')
    #
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader), start=1):
            #
            obs_bbox = batch['input_x'].to(device)
            pred_bbox_gt = batch['target_y'].to(device)
            img_path = batch['cur_image_file']
            
            #
            #obs_img_feats = batch['obs_img_feat'].to(device)
            obs_img_feats = batch['obs_pose_feat'].to(device)
            intention_binary = batch['intention_binary'].to(device)
            obsLast_ped_look = batch['obs_ped_look'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = batch['obs_body_ori'][:, -1].reshape([-1, 1]).to(device)
            obsLast_body_ori = obsLast_body_ori * math.pi / 180.0
            
            ########
            if dataset_name=='PIE':
                #
                obs_sensor_speed = batch['obs_obd_speed'].to(device)
                obs_sensor_heading_angle = batch['obs_heading_angle'].to(device)
                obs_ego = torch.cat([obs_sensor_speed, obs_sensor_heading_angle], dim=-1)
            else:
                #
                obs_ego = batch['obs_ego_act'].to(device)
            
            
            # timing (start)
            torch.cuda.synchronize()
            t1 = time.time()
            
            # inference
            pred_crossing, pred_traj = model.predict(obs_img_feats, obsLast_ped_look, obsLast_body_ori, 
                                                     obs_bbox=obs_bbox, obs_ego=obs_ego)
            
            # timing (end)
            torch.cuda.synchronize()
            t2 = time.time()
            time_currIter = t2 - t1
            #
            avgTime_iter += time_currIter
            total_iter += 1
            
            #print(time_currIter)
            
            #
            pred_crossing_list.append(pred_crossing.cpu().detach().numpy())
            gt_crossing_list.append(intention_binary.cpu().detach().numpy())
            
            
            ######## transfer back to global coordinates
            ret = post_process(cfg, obs_bbox, pred_bbox_gt, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None)
            X_global, y_global, _, pred_traj, _, _ = ret
            all_img_paths.extend(img_path)
            all_X_globals.append(X_global)
            #all_pred_goals.append(pred_goal)
            all_pred_trajs.append(pred_traj)
            #all_gt_goals.append(y_global[:, -1])
            all_gt_trajs.append(y_global)
            all_timesteps.append(batch['timestep'].numpy())
            
            
            if traj_viz:
                #### my modification (i think i should visualize all batches!)
                traj_viz_save_root = cfg.VIZ.TRAJ_VIZ_SAVE_ROOT
                #
                curr_pred_crossing_np = np.round(pred_crossing.cpu().detach().numpy())
                curr_gt_crossing_np = intention_binary.cpu().detach().numpy()
                #
                my_viz_results(viz, X_global, y_global, pred_traj, img_path, traj_viz_save_root, 
                               curr_iter=iters-1, test_batch_size=X_global.shape[0], 
                               bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, 
                               pred_crossing=curr_pred_crossing_np, gt_crossing=curr_gt_crossing_np)
    
    
            
            
            
    ######## print the average time of inferencing.
    avgTime_iter /= total_iter
    print('the average time of inferencing is: '+str(avgTime_iter)+',    total iter number is:'+str(total_iter))
    
    ######## Crossing Intention: compute the metric (accuracy, f1-score)
    pred_crossing_np = np.concatenate(pred_crossing_list, axis=0)
    gt_crossing_np = np.concatenate(gt_crossing_list, axis=0)
    acc_crossing = accuracy_score(gt_crossing_np, np.round(pred_crossing_np))
    f1_crossing = f1_score(gt_crossing_np, np.round(pred_crossing_np), average='micro')
    
    print('accuracy of crossing: {:.4f}, f1 score of crossing: {:.4f}'.format(acc_crossing, f1_crossing))
    
    
    ######## Trajectory:
    all_X_globals = np.concatenate(all_X_globals, axis=0)
    #all_pred_goals = np.concatenate(all_pred_goals, axis=0)
    all_pred_trajs = np.concatenate(all_pred_trajs, axis=0)
    #all_gt_goals = np.concatenate(all_gt_goals, axis=0)
    all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)
    all_timesteps = np.concatenate(all_timesteps, axis=0)
    mode = 'bbox' if all_gt_trajs.shape[-1] == 4 else 'point'
    eval_results = evaluate_unimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=None, bbox_type=cfg.DATASET.BBOX_TYPE)
    
#     ######## just set the sample num to 1, we can also use the 'evaluate_multimodal(.)' to evaluate the result.
#     all_pred_trajs = np.expand_dims(all_pred_trajs, axis=2)
#     eval_results = evaluate_multimodal(all_pred_trajs, all_gt_trajs, mode=mode, distribution=None, bbox_type=cfg.DATASET.BBOX_TYPE)
    
    for key, value in eval_results.items():
        info = "Testing prediction {}:{}".format(key, str(np.around(value, decimals=3)))
        if hasattr(logger, 'log_values'):
            logger.info(info)
        else:
            print(info)
        
        # my adding
        if key=='C-FDE':
            c_fde_inference = value
        if key=='C-ADE(1.5s)':
            c_ade_inference = value
        if key=='ADE(0.5s)':
            ade_5_inference = value
        if key=='ADE(1.0s)':
            ade_10_inference = value
        if key=='ADE(1.5s)':
            ade_15_inference = value
    
    if hasattr(logger, 'log_values'):
        logger.log_values(eval_results)
    
    if test_mode:
        # save inputs, redictions and targets for test mode
        outputs = {'img_path': all_img_paths, 'X_global': all_X_globals, 'timestep': all_timesteps,
                   'pred_trajs': all_pred_trajs, 'gt_trajs':all_gt_trajs,'distributions':None}
        
        if not os.path.exists(cfg.OUT_DIR):
            os.makedirs(cfg.OUT_DIR)
        output_file = os.path.join(cfg.OUT_DIR, '{}_{}.pkl'.format(cfg.MODEL.LATENT_DIST, cfg.DATASET.NAME))
        print("Writing outputs to: ", output_file)
        pkl.dump(outputs, open(output_file,'wb'))
    
    
    
    
    # visdom visualization
    if (visdom_viz is not None) and (t is not None):
        visdom_viz.line([[acc_crossing, f1_crossing]], [t], win='ci_inference', update='append')
        visdom_viz.line([[c_ade_inference, c_fde_inference]], [t], win='c_ade_fde_inference', update='append')
        visdom_viz.line([[ade_5_inference, ade_10_inference, ade_15_inference]], [t], win='ade_inference', update='append')
    
    
    return acc_crossing, f1_crossing, c_ade_inference, c_fde_inference









