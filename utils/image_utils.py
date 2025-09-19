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
import math
import cv2
from scipy import stats
import torch.nn as nn
import torch
import numpy as np
import matplotlib
import torch.nn.functional as F

from utils.loss_utils import cal_gradient

def visualize_depth(depth, near=0.2, far=13):
    depth = depth[0].detach().cpu().numpy()
    colormap = matplotlib.colormaps['turbo']
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = near if near else depth.min()
    far = far if far else depth.max()
    near -= eps
    far += eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]

    out_depth = np.clip(np.nan_to_num(vis), 0., 1.)
    return torch.from_numpy(out_depth).float().cuda().permute(2, 0, 1)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# def get_hf_image(rgb):
#     ycrcb = rgb2ycrcb(rgb)
#     rgb_max, _ = rgb.max(dim=0)
#     rgb_min, _ = rgb.min(dim=0)
#     res = rgb_max - rgb_min
#     hf_image =torch.stack([res, ycrcb[1], ycrcb[2]], dim=0)
#     return hf_image


def rgb_to_srgb(rgb):
    srgb = torch.where(rgb > 0.0031308, torch.pow(1.055 * rgb, 1.0 / 2.4) - 0.055, rgb * 12.92)
    return srgb
    ret = torch.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = torch.pow(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def srgb_to_rgb(srgb):
    rgb = torch.where(srgb > 0.04045, torch.pow((srgb + 0.055) / 1.055, 2.4), srgb / 12.92)
    return rgb
    ret = torch.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = torch.pow((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb2ycbcr(rgb):
    """
    rgb [3, h, w] 0-1
    """
    r, g, b = rgb
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.stack([y, cb, cr], dim=0)


def ycrcb2rgb(ycbcr): # 用于可视化
    y, cb, cr = ycbcr
    b = (cb - 0.5) * 1. / 0.564 + y
    r = (cr - 0.5) * 1. / 0.713 + y
    g = 1. / 0.587 * (y - 0.299 * r - 0.114 * b)
    return torch.stack([r, g, b], dim=0)
    
    
def get_hf_image(rgb):
    ycbcr = rgb2ycbcr(rgb)
    rgb_max, _ = torch.max(rgb, dim=0)
    rgb_min, _ = torch.min(rgb, dim=0)
    res = rgb_max - rgb_min
    hf_image =torch.stack([res, ycbcr[1], ycbcr[2]], dim=0)
    return hf_image


def rgb2ycbcr_np(rgb):
    """
    rgb [h, w, 3] 0-1
    """
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return np.stack([y, cb, cr], axis=-1)

def ycbcr2rgb_np(ycbcr):
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    b = (cb - 0.5) * 1. / 0.564 + y
    r = (cr - 0.5) * 1. / 0.713 + y
    g = 1. / 0.587 * (y - 0.299 * r - 0.114 * b)
    return np.stack([r, g, b], axis=-1)

def get_hf_image_np(image): # 数据集准备时用
    """
    image [h w 3] 0-1
    return [h w 3] 0-1
    """
    ycbcr = rgb2ycbcr_np(image)
    rgb_max = image.max(axis=-1)
    rgb_min = image.min(axis=-1)
    res = rgb_max - rgb_min
    cb = ycbcr[...,1]
    cr = ycbcr[...,2]
    hf_image = np.stack([res, cb, cr], axis=-1)
    return hf_image
    
def get_sf_image(image): # 数据集准备时用
    # image [h w 3] 0-255 np int8
    # return [h w 3] 0-1  np
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = torch.from_numpy(image).float() #  torch is faster than np
    image[image == 0] = 1
    div = torch.pow(torch.prod(image, dim=-1), 1/3)
    rho = torch.log1p((image / div[..., None]) - 1)
    U = torch.tensor([[1/math.sqrt(2), -1/math.sqrt(2),                 0],
                      [1/math.sqrt(6),  1/math.sqrt(6),  -2/math.sqrt(6)]])
    chi = torch.matmul(rho, U.T)
    angles = torch.linspace(0, torch.pi, 181)
    e_t = torch.stack((torch.cos(angles), torch.sin(angles)))
    Y = torch.matmul(chi, e_t)
    nel = image.shape[0] * image.shape[1]
    entropy = []
    muhat, sigma, muci, sigmaci = normfit(Y)
    comp1 = muhat - sigmaci[0]
    comp2 = muhat + sigmaci[1]
    for i in range(181):
        temp = Y[..., i][(Y[..., i] > comp1[i]) & (Y[..., i] < comp2[i])]
        
        bw = (3.5 * torch.std(temp)) * ((nel) ** (-1.0 / 3))
        
        nbins = torch.ceil((temp.max() - temp.min()) / bw).int()
        # print(nbins)
        hist = torch.histc(temp, bins=nbins, min=temp.min(), max=temp.max())
        
        hist = hist[hist != 0]
        hist = hist.float() / torch.sum(hist)
        entropy.append(-1 * torch.sum(torch.multiply(hist, torch.log2(hist))).item())
        # break
    angle = entropy.index(min(entropy))    
    # print(angle)
    e=np.array([[-1*math.sin(angle*math.pi/180.0)],
            [math.cos(angle*math.pi/180.0)]])
    e_t=np.array([[math.cos(angle*math.pi/180.0)],
                [math.sin(angle*math.pi/180.0)]])
    P_theta = np.ma.divide(np.dot(e_t, e_t.T), np.linalg.norm(e))
    chi_theta = chi.numpy().dot(P_theta)
    rho_estim = chi_theta.dot(U.numpy())
    mean_estim = np.ma.exp(rho_estim)                                                   
    estim = np.zeros_like(mean_estim, dtype=np.float64)
    estim[:,:,0] = np.divide(mean_estim[:,:,0], np.sum(mean_estim, axis=2))
    estim[:,:,1] = np.divide(mean_estim[:,:,1], np.sum(mean_estim, axis=2))
    estim[:,:,2] = np.divide(mean_estim[:,:,2], np.sum(mean_estim, axis=2))
    return estim


def normfit(data, confidence=0.9):
    m = torch.mean(data, dim=(0,1))
    se = torch.std(data, dim=(0,1)) / torch.sqrt(torch.tensor(data.size(0), dtype=torch.float64))
    h = se * torch.tensor(stats.t.ppf((1 + confidence) / 2., data.size(0) - 1))
    
    var = torch.var(data, dim=(0,1), unbiased=True)
    
    varCI_upper = var * (data.size(0) - 1) / torch.tensor(stats.chi2.ppf((1 - confidence) / 2, data.size(0) - 1), dtype=torch.float64)
    varCI_lower = var * (data.size(0) - 1) / torch.tensor(stats.chi2.ppf(1 - (1 - confidence) / 2, data.size(0) - 1), dtype=torch.float64)
    
    sigma = torch.sqrt(var)
    
    sigmaCI_lower = torch.sqrt(varCI_lower)
    sigmaCI_upper = torch.sqrt(varCI_upper)
    return m, sigma, [m - h, m + h], [sigmaCI_lower, sigmaCI_upper]

def get_chromaticity_image(image):#色度
    # [3 h w] 
    return image / torch.sum(image, dim=0, keepdim=True)

def get_laplacian_image_np(image):#拉普拉斯变换,Laplacian算子通过计算图像的二次空间导数来检测边缘
    # [h w, 3]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian) / 255
    laplacian = laplacian[..., None]
    return laplacian

def cal_local_weights(chorm, alpha=60):
    chrom_grad = cal_gradient(chorm.mean(dim=0, keepdim=True).unsqueeze(0)).squeeze(0)
    weight = torch.exp(- alpha * chrom_grad)
    return weight


def adjust_image_for_display(image, trans2srgb=True, rescale=True):
    MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837

    vis = image.clone()
    if rescale:
        s = np.percentile(image.cpu().numpy(), 99.9).item()
        print(s)
        if s > MAX_SRGB:
            vis = vis / s * MAX_SRGB

    vis = torch.clamp(vis, min=0)
    # vis[vis < 0] = 0
    if trans2srgb:
        vis[vis > MAX_SRGB] = MAX_SRGB
        vis = rgb_to_srgb(vis)
    return vis



def rgb2xyz_np(rgb):
    # 将RGB值转换为XYZ空间
    # rgb < 1
    # rgb [N, 3]
    if rgb.max() > 10:
        rgb = rgb / 255.
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
 
    xyz = rgb@m.T
    return xyz


def xyz2lab_np(xyz, normalize=True):
    # 将XYZ值转换为Lab空间
    xyz_ref = np.array([0.950456, 1.0, 1.088754])
    
    xyz_ratio = xyz / xyz_ref
    xyz_ratio = np.where(xyz_ratio > 0.008856, xyz_ratio ** (1/3), (903.3 * xyz_ratio + 16) / 116)
    
    lab = np.zeros_like(xyz)
    lab[..., 0] = np.clip(116 * xyz_ratio[..., 1] - 16, 0, 100)
    lab[..., 1] = (xyz_ratio[..., 0] - xyz_ratio[..., 1]) * 500
    lab[..., 2] = (xyz_ratio[..., 1] - xyz_ratio[..., 2]) * 200
    if normalize:
        lab[..., 0] /= 100
        lab[..., 1] = (lab[..., 1] + 128) / 255
        lab[..., 2] = (lab[..., 2] + 128) / 255

    return lab


def rgb2lab_np(rgb):
    xyz = rgb2xyz_np(rgb)
    lab = xyz2lab_np(xyz)
    return lab



def rgb2xyz(rgb):
    # 将RGB值转换为XYZ空间
    # rgb < 1
    # rgb [N, 3]
    # if rgb.max() > 10:
    #     rgb = rgb / 255.
    if rgb.max() > 10:
        rgb = rgb / 255.
    rgb = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=rgb.device)
    
    xyz = rgb@m.T
    return xyz


def xyz2lab(xyz, normalize=True):
    xyz_ref = torch.tensor([0.950456, 1.0, 1.088754], device=xyz.device)
    
    xyz_ratio = xyz / xyz_ref
    xyz_ratio = torch.where(xyz_ratio > 0.008856, xyz_ratio ** (1/3), (903.3 * xyz_ratio + 16) / 116)
    
    l = torch.clamp(116 * xyz_ratio[..., 1] - 16, 0,  100)
    a = (xyz_ratio[..., 0] - xyz_ratio[..., 1]) * 500
    b = (xyz_ratio[..., 1] - xyz_ratio[..., 2]) * 200
    
    if normalize:
        l = l / 100
        a = (a + 128) / 255
        b = (b + 128) / 255

    lab = torch.stack([l, a, b], dim=-1)
    
    return lab


def rgb2lab(rgb):
    xyz = rgb2xyz(rgb)
    lab = xyz2lab(xyz)
    return lab


def lab2flab(lab, scale=0.3):
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    return torch.stack([l*scale, a, b], dim=-1)

def lab2flab_np(lab, scale=0.3):
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    return np.stack([l*scale, a, b], axis=-1)