#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def cal_laplacian(data):
    """
    data: [1, c, H, W]
    """
    kernel = [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(data.device)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    laplacian = F.conv2d(data, weight, padding='same')
    return laplacian


def cal_gradient(data, p=1):
    """
    data: [1, C, H, W]，看这里调用这个函数的用法，似乎传入的数据都在C维度做了均值化，似乎形状都为[1,1,H,W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    
    # gradient = torch.abs(grad_x) + torch.abs(grad_y)
    grad = torch.cat([grad_x, grad_y], dim=-3)#[1,2,H,W]
    gradient = torch.norm(grad, p=p, dim=-3, keepdim=True)
    return gradient

def cal_gradient_xy(data):
    """
    data: [1, C, H, W]
    """
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(data.device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(data.device)

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    grad_x = F.conv2d(data, weight_x, padding='same')
    grad_y = F.conv2d(data, weight_y, padding='same')
    # gradient = torch.abs(grad_x) + torch.abs(grad_y)

    return grad_x, grad_y


def bilateral_sparsity_loss(data, image, mask, coff):#双边稀疏损失，保持图像边缘细节的同时进行平滑处理
    """
    image: [C, H, W]
    data: [C, H, W]
    mask: [C, H, W]
    """
    rgb_grad = cal_gradient(image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]
    data_grad = cal_gradient(data.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)  # [1, H, W]

    smooth_loss = (data_grad * (-rgb_grad * coff).exp() * mask).mean()

    return smooth_loss


def compute_residual_loss(residual):
    return torch.mean(residual**2)


# def get_separation_loss(img1, img2):
#     loss = 0.0
#     grad1 = cal_gradient(img1.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#     grad2 = cal_gradient(img2.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#     lambda1 = torch.sqrt(torch.norm(grad1)/(torch.norm(grad2)))
#     lambda2 = torch.sqrt(torch.norm(grad2)/(torch.norm(grad1)))
#     alpha1 = torch.tanh(lambda1 * grad1)
#     alpha2 = torch.tanh(lambda2 * grad2)
#     loss += torch.norm(alpha1 * alpha2)
#     return loss
    
    
# def get_separation_loss(img1, img2, level=1):
#     loss = 0.0
    
#     for i in range(level):
#         grad1 = cal_gradient(img1.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#         grad2 = cal_gradient(img2.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#         lambda1 = torch.sqrt(torch.norm(grad1)/(torch.norm(grad2)))
#         lambda2 = torch.sqrt(torch.norm(grad2)/(torch.norm(grad1)))
#         alpha1 = torch.tanh(lambda1 * grad1)
#         alpha2 = torch.tanh(lambda2 * grad2)
#         loss += torch.norm(alpha1 * alpha2)
#         # img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear', align_corners=False)
#         # img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear', align_corners=False)

#     return loss

def light_smooth_loss(light, normal, normal_thrld=0.05):
    light_mag = torch.linalg.norm(light, ord=2, dim=0, keepdim=True)
    light_dir = light / light_mag.clamp(min=1e-6)
    normal_grad = cal_gradient(torch.abs(normal).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    flatten_mask = torch.where(normal_grad < normal_thrld, True, False)
    # print(torch.sum(flatten_mask).item()/(flatten_mask.shape[1]*flatten_mask.shape[2]))
    light_grad = cal_gradient(light_mag.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    light_grad_valid = light_grad[flatten_mask]
    mag_loss = light_grad_valid.mean()
    dir_grad = cal_gradient(torch.abs(light_dir).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    dir_loss = dir_grad.mean()
    return mag_loss, dir_loss
    
    
def get_depth_loss(depth_pred, depth_gt, mask_gt):
    loss_list = []
    for pred, gt, mask in zip(depth_pred, depth_gt, mask_gt):
        log_pred = torch.log(pred[mask])
        log_target = torch.log(gt[mask])
        alpha = (log_target - log_pred).sum()/mask.sum()
        log_diff = torch.abs((log_pred - log_target + alpha))
        d = log_diff.sum()/mask.sum()
        loss_list.append(d)

    return torch.stack(loss_list, 0).mean()


# def get_sparsity_weight(gt_rgb, gt_normal, gt_lab, rgb_scale=0.3, normal_scale=1, a=60, b=8):
#     lab_feature = torch.stack([gt_lab[0]*0.3, gt_lab[1], gt_lab[2]])
#     # lab_norm = torch.norm(lab_feature,p=1, dim=0, keepdim=True)
#     # lab_grad = cal_gradient((lab_norm**1).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#     # rgb_grad = cal_gradient(gt_rgb.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
#     lab_grad = cal_gradient(lab_feature.mean(0, keepdim=True).unsqueeze(0), p=2).squeeze(0) ** 2
#     rgb_grad = cal_gradient(gt_rgb.mean(0, keepdim=True).unsqueeze(0), p=2).squeeze(0) ** 2
#     # lab_norm = torch.norm(lab_feature,p=2, dim=0, keepdim=True)
#     # lab_grad = cal_gradient((lab_norm**2).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
#     # rgb_grad = cal_gradient(gt_rgb.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
#     lab_rgb_grad = torch.where(lab_grad > rgb_scale * rgb_grad , lab_grad, rgb_scale * rgb_grad)
    
#     return torch.exp(-60 * lab_rgb_grad)
    
#     # normal = (gt_normal + 1) / 2 # [-1, 1] -> [0, 1]
#     normal_grad = cal_gradient(gt_normal.abs().mean(0, keepdim=True).unsqueeze(0), p=2).squeeze(0) ** 2
#     normal_grad = torch.where(normal_grad > 0.5, normal_grad, 0)
    
#     lab_rgb_normal_grad = torch.where(lab_rgb_grad > normal_scale * normal_grad, lab_rgb_grad, normal_scale * normal_grad)
    
#     # 1 / (1 + exp(ax-b))
#     lab_rgb_weight = 1 / (1 + torch.exp(a * lab_rgb_normal_grad - b))
    
    # return lab_rgb_weight


def get_sparsity_weight(gt_image, gt_normal, lab, rgb_grad_scale=0.3, normal_grad_scale=0.3, a=80, b=16):
    lab_feature = torch.stack([lab[0]*0.3, lab[1], lab[2]])
    # lab_norm = torch.norm(lab_feature,p=1, dim=0, keepdim=True)
    # lab_grad = cal_gradient((lab_norm**1).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    lab_grad = cal_gradient((lab_feature).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    rgb_grad = cal_gradient(gt_image.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    lab_rgb_grad = torch.where(lab_grad > rgb_grad_scale * rgb_grad , lab_grad, rgb_grad_scale * rgb_grad)#torch.where(condition, x, y)根据condition选择x或者y
    
    normal = (gt_normal + 1) / 2 # [-1, 1] -> [0, 1]
    normal_grad = cal_gradient(normal.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    
    lab_rgb_normal_grad = torch.where(lab_rgb_grad > normal_grad_scale * normal_grad, lab_rgb_grad, normal_grad_scale * normal_grad)
    
    # 1 / (1 + exp(ax-b))
    lab_rgb_weight = 1 / (1 + torch.exp(a * lab_rgb_normal_grad - b))
    
    return lab_rgb_weight


def get_smooth_weight(gt_depth, gt_normal, gt_lab, depth_threshold=0.1, normal_threshold=0.2):
    lab_feature = torch.stack([gt_lab[0]*0.3, gt_lab[1], gt_lab[2]])
    # lab_norm = torch.norm(lab_feature,p=2, dim=0, keepdim=True)
    # lab_grad = cal_gradient((lab_norm**2).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    lab_grad = cal_gradient((lab_feature).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)

    #TODO 为什么直接用lab_grad作为权重，而不是下面这个weight？ 答：这个depth_grad没用，本文的normal_grad在调用get_smooth_weight之后再计算的，实现这个代码的论文似乎有depth_grad，但是本文不用depth就有此实现上的区别
    # depth_grad = cal_gradient(gt_depth.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    # normal_grad = cal_gradient(gt_normal.abs().mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    # smooth_mask = torch.where(normal_grad < normal_threshold, True, False)
    # weight = normal_grad[smooth_mask]
    
    return lab_grad
    
    

    
class LaplaceFilter_5D(nn.Module):
    def __init__(self):
        super(LaplaceFilter_5D, self).__init__()
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        edge = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ])
        edge_k = edge
        edge_k = torch.from_numpy(edge_k).to(torch.float32).view(1, 1, 5, 5)
        self.edge_conv.weight = nn.Parameter(edge_k, requires_grad=False)

        if True:
            self.mask_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
            mask_k = np.array([
                [0, 0, 0.077, 0, 0],
                [0, 0.077, 0.077, 0.077, 0],
                [0.077, 0.077, 0.077, 0.077, 0.077],
                [0, 0.077, 0.077, 0.077, 0],
                [0, 0, 0.077, 0, 0]
            ])
            mask_k = torch.from_numpy(mask_k).to(torch.float32).view(1, 1, 5, 5)
            self.mask_conv.weight = nn.Parameter(mask_k, requires_grad=False)

        for param in self.parameters():
            param.requires_grad = False

    def apply_laplace_filter(self, x, mask=None):
        out = self.edge_conv(x)
        if mask is not None:
            out_mask = self.mask_conv(mask)
            out_mask[out_mask < 0.95] = 0
            out_mask[out_mask >= 0.95] = 1
            out = torch.mul(out, out_mask)
        else:
            out_mask = None
        return out, out_mask

    def forward(self, x, mask=None):
        out, out_mask = self.apply_laplace_filter(x[:, 0:1, :, :], mask[:, 0:1, :, :] if mask is not None else None)
        for idx in range(1, x.size(1)):
            d_out, d_out_mask = self.apply_laplace_filter(x[:, idx:idx+1, :, :],
                                                          mask[:, idx:idx+1, :, :] if mask is not None else None)
            out = torch.cat((out, d_out), 1)
            if d_out_mask is not None:
                out_mask = torch.cat((out_mask, d_out_mask), 1)
