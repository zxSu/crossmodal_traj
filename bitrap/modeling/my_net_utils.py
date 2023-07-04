import torch
import torch.nn as nn
import math


def make_mlp(dim_list, activation='relu', batch_norm=False, bias=False, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)




def make_embedding(dim_in, dim_out, activation='relu', batch_norm=True, bias=False, dropout=0):
    layers = []
    layers.append(nn.Linear(dim_in, dim_out, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm1d(dim_out))
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)




def rmse_loss(pred_traj, gt_traj):
    traj_rmse = torch.sqrt(torch.sum((pred_traj - gt_traj)**2, dim=-1)).sum(dim=1)
    final_loss = traj_rmse.mean()
    
    return final_loss



def exp_rmse_loss(pred_traj, pred_traj_gt):
    # (1) generate the matrix for 'exponential_term'
    num_time = pred_traj_gt.shape[1]
    time_linspace = torch.linspace(0, num_time-1, steps=num_time, device='cuda', requires_grad=False)
    r = 60
    exp_term = torch.exp(time_linspace/r).reshape([1, -1])
    num_traj = pred_traj_gt.shape[0]
    exp_term_repeat = exp_term.repeat(num_traj, 1)    # size: [num_traj, num_time]
    #
    traj_rmse = torch.sqrt(torch.sum((pred_traj - pred_traj_gt)**2, dim=-1))
    #
    exp_traj_rmse = traj_rmse * exp_term_repeat
    exp_traj_rmse = torch.sum(exp_traj_rmse, dim=1)
    #
    return torch.mean(exp_traj_rmse)





def cvae_loss(pred_trajs, gt_traj, kld, kld_weight=1.0):
    #
    K = pred_trajs.shape[2]
    gt_trajs = gt_traj.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    #
    final_loss = loss_traj + kld_weight * kld
    
    return final_loss




def exp_cvae_loss(pred_trajs, pred_traj_gt, kld, kld_weight=1.0):
    # (1) generate the matrix for 'exponential_term'
    num_sample = pred_trajs.shape[2]
    num_time = pred_traj_gt.shape[1]
    time_linspace = torch.linspace(0, num_time-1, steps=num_time, device='cuda', requires_grad=False)
    r = 60
    exp_term = torch.exp(time_linspace/r).reshape([1, -1, 1])
    num_traj = pred_traj_gt.shape[0]
    exp_term_repeat = exp_term.repeat(num_traj, 1, num_sample)    # size: [num_traj, num_time, num_sample]
    #
    gt_trajs = pred_traj_gt.unsqueeze(2).repeat(1, 1, num_sample, 1)
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs)**2, dim=-1))
    exp_traj_rmse = traj_rmse * exp_term_repeat
    exp_traj_rmse = torch.sum(exp_traj_rmse, dim=1)
    #
    best_idx = torch.argmin(exp_traj_rmse, dim=1)
    loss_traj = exp_traj_rmse[range(len(best_idx)), best_idx].mean()
    #
    final_loss = loss_traj + kld_weight * kld
    
    return final_loss





def cvae_loss_v2(pred_trajs, pred_goals, gt_traj, kld, kld_weight=1.0):
    #
    K = pred_trajs.shape[2]
    gt_trajs = gt_traj.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred_trajs[:, :-1, :, :] - gt_trajs[:, :-1, :, :])**2, dim=-1)).sum(dim=1)
    goal_rmse = torch.sqrt(torch.sum((pred_goals - gt_trajs[:, -1, :, :])**2, dim=-1))
    # bom based on 'goal' or 'traj'?
    best_idx = torch.argmin(goal_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    loss_goal = goal_rmse[range(len(best_idx)), best_idx].mean()
    
    #
    final_loss = loss_traj + loss_goal + kld_weight * kld
    
    return final_loss








def bom_l2_loss(pred_trajs, gt_traj):
    #
    K = pred_trajs.shape[2]
    gt_trajs = gt_traj.unsqueeze(2).repeat(1, 1, K, 1)
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs)**2, dim=-1)).sum(dim=1)
    best_idx = torch.argmin(traj_rmse, dim=1)
    loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
    
    return loss_traj





######## 2023; related to uncertainty
def regression_uncertainty_loss(pred_trajs, gt_trajs, tao=1):
    
    # mean_errors = ( (gt_trajs.unsqueeze(dim=2) - pred_trajs)**2 ).mean(dim=2)
    # traj_loss = (1.0 / pred_trajs.shape[0]) * mean_errors.sum()
    #
    # return traj_loss
    
    # traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs.unsqueeze(dim=2))**2, dim=-1)).sum(dim=1)
    # final_loss = traj_rmse.mean()
    
    #
    num_sample = pred_trajs.shape[2]
    d = pred_trajs.shape[3]
    traj_rmse = -0.5 * tao * torch.sqrt(torch.sum((pred_trajs - gt_trajs.unsqueeze(dim=2))**2, dim=-1)).sum(dim=1)    # size: [batch_size, num_sample]
    traj_log_likelihood = torch.log(torch.sum(torch.exp(traj_rmse), dim=1)) - math.log(num_sample) - 0.5 * d * math.log(2*math.pi) - 0.5 * math.log(1.0/tao)    # size: [batch_size, ]
    final_loss = (- traj_log_likelihood / tao).mean()
    
    return final_loss




def regression_uncertainty_loss_complex(pred_trajs, gt_trajs, var_trajs):
    
    s = torch.log(var_trajs.squeeze(dim=2))
    term_1 = 0.5 * torch.exp(-s) * torch.sqrt(torch.sum((pred_trajs - gt_trajs.unsqueeze(dim=2))**2, dim=-1)).sum(dim=1)    # size: [batch_size, num_sample]
    term_2 = 0.5 * s    # size: [batch_size, num_sample]
    final_loss = (term_1 + term_2).mean()
    
    return final_loss



######## strictly follow the equation of 'negative log-likelihood'
def regression_uncertainty_loss_complex_v2(pred_trajs, gt_trajs, var_trajs):
    
    num_sample = pred_trajs.shape[2]
    d = pred_trajs.shape[3]
    s = var_trajs.squeeze(dim=2)
    traj_rmse = -0.5 * (1.0 / s) * torch.sqrt(torch.sum((pred_trajs - gt_trajs.unsqueeze(dim=2))**2, dim=-1)).sum(dim=1)    # size: [batch_size, num_sample]
    term_1 = - math.log(num_sample) - 0.5 * d * math.log(2*math.pi)
    term_2 = (1.0 / s.pow(0.5)) * torch.exp(traj_rmse)
    term_3 = torch.log(term_2.sum(dim=1))
    final_loss = - (term_1 + term_3).mean()
    
    return final_loss






def regression_uncertainty_loss_oneSample(pred_trajs, gt_trajs):
    
    traj_rmse = torch.sqrt(torch.sum((pred_trajs - gt_trajs)**2, dim=-1)).sum(dim=1)    # size: [batch_size, ]
    final_loss = traj_rmse.mean()
    
    return final_loss




def regression_uncertainty_loss_complex_oneSample(pred_trajs, gt_trajs, var_trajs):
    
    s = torch.log(var_trajs.squeeze(dim=1))    # size: [batch_size, ]
    term_1 = 0.5 * torch.exp(-s) * torch.sqrt(torch.sum((pred_trajs - gt_trajs)**2, dim=-1)).sum(dim=1)    # size: [batch_size, ]
    term_2 = 0.5 * s    # size: [batch_size, ]
    final_loss = (term_1 + term_2).mean()
    
    return final_loss



def classification_uncertainty_loss_oneSample(pred, gt):
    
    batch_size = pred.shape[0]
    # apply the enhanced 'logsumexp' trick (simplest one)
    loss_raw = pred[range(batch_size), gt] - torch.log( torch.sum(torch.exp(pred), dim=1) )    # size: [batch_size, ]
    # # apply the enhanced 'logsumexp' trick (solve the problem of overflow and underflow)
    # c = pred.max(dim=1)[0]    # size: [batch_size, ]
    # loss_raw = pred[range(batch_size), gt] - torch.log( torch.sum(torch.exp(pred - c.unsqueeze(dim=1)), dim=1) ) - c    # size: [batch_size, ]
    #
    loss_final = - loss_raw.mean()
    
    return loss_final








def classification_uncertainty_loss(pred, gt):
    
    num_sample = pred.shape[1]
    batch_size = pred.shape[0]
    loss_raw_list = []
    for i in range(num_sample):
        pred_currSample = pred[:, i, :]    # size: [batch_size, 3]
        # apply the enhanced 'logsumexp' trick (simplest one)
        curr_loss_raw = pred_currSample[range(batch_size), gt] - torch.log( torch.sum(torch.exp(pred_currSample), dim=1) )    # size: [batch_size, ]
        # # apply the enhanced 'logsumexp' trick (solve the problem of overflow and underflow)
        # curr_c = pred_currSample.max(dim=1)[0]    # size: [batch_size, ]
        # curr_loss_raw = pred_currSample[range(batch_size), gt] - torch.log( torch.sum(torch.exp(pred_currSample - curr_c.unsqueeze(dim=1)), dim=1) ) - curr_c    # size: [batch_size, ]
        loss_raw_list.append(curr_loss_raw)
    #
    loss_raw_all = torch.stack(loss_raw_list, dim=1)    # size: [batch_size, num_sample]
    # compute the average log value from many log values
    loss_final = - (1.0 / batch_size) * torch.log( torch.exp(loss_raw_all).sum(dim=1) * (1.0 /  num_sample) ).sum()
    
    return loss_final




def classification_uncertainty_loss_v2(loss_unit_fn, pred, gt):
    
    batch_size = pred.shape[0]
    num_sample = pred.shape[1]
    loss_list = []
    for i in range(num_sample):
        pred_currSample = pred[:, i, :]    # size: [batch_size, 3]
        loss_list_currSample = []
        for j in range(batch_size):
            pred_curr = pred_currSample[j, :].unsqueeze(dim=0)    # size: [1, 3]
            gt_curr = gt[j].unsqueeze(dim=0)    # size: [1, 1]
            curr_loss = loss_unit_fn(pred_curr, gt_curr)
            loss_list_currSample.append(curr_loss)
        #
        losses_currSample = torch.stack(loss_list_currSample, dim=0)    # size: [batch_size, ]
        loss_list.append(losses_currSample)
    #
    loss_all = torch.stack(loss_list, dim=1)    # size: [batch_size, num_sample]
    #
    loss_final = (1.0 / batch_size) * loss_all.mean(dim=1).sum()
    
    return loss_final





    
    