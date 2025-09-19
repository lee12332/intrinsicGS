import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import imageio.v2 as imageio
from utils.image_utils import adjust_image_for_display, get_sf_image, get_hf_image_np, get_hf_image

# def rgb2ycbcr_np(rgb):
#     """
#     rgb [h, w, 3] 0-1
#     """
#     r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     cb = (b - y) * 0.564 + 0.5
#     cr = (r - y) * 0.713 + 0.5
#     return np.stack([y, cb, cr], axis=-1)


# def ycbcr2rgb_np(ycbcr):
#     y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
#     b = (cb - 0.5) * 1. / 0.564 + y
#     r = (cr - 0.5) * 1. / 0.713 + y
#     g = 1. / 0.587 * (y - 0.299 * r - 0.114 * b)
#     return np.stack([r, g, b], axis=-1)


# def get_hf_image_np(image):
#     """
#     image [h w 3] 0-1
#     """
#     ycbcr = rgb2ycbcr_np(image)
#     rgb_max = image.max(axis=-1)
#     rgb_min = image.min(axis=-1)
#     res = rgb_max - rgb_min
#     cb = ycbcr[...,1]
#     cr = ycbcr[...,2]
#     hf_image = np.stack([res, cb, cr], axis=-1)
#     return hf_image




# img = imageio.imread('/home/lhy/research/intrinsic-gs/datasets/intrinsic/50/replica/office_0/images/rgb_55.png')
# # img = img / 255.0
# print(img.dtype)
# sf_image = get_sf_image(img)

# print(sf_image.dtype)

# img = img /255
# print(img.dtype)
# hf_image = get_hf_image_np(img)
# print(hf_image.dtype)

# img_torch = torch.from_numpy(img)
# print(img_torch.dtype)
# img_torch = img_torch.float()
# print(img_torch.dtype)


# # hf_img = get_hf_image_np(img)
# # show_img = ycbcr2rgb_np(hf_img)
# # show_img = np.clip(show_img, 0, 1)
# # show_img = (show_img * 255).astype(np.uint8)
# # imageio.imwrite('hf_img.png', show_img)

# sf_image = get_sf_image(img)
# show_img = (sf_image * 255).astype(np.uint8)
# imageio.imwrite('sf_img.png', show_img)

# def grad_separation_loss(img1, img2, level=3):
    
#     def compute_gradient(img):
#         gradx = img[:, 1:, :] - img[:, :-1, :]
#         grady = img[:, :, 1:] - img[:, :, :-1]
#         return gradx, grady
    
#     gradx_loss = []
#     grady_loss = []
    
#     for i in range(level):
#         gradx1, grady1 = compute_gradient(img1)
#         gradx2, grady2 = compute_gradient(img2)
#         alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
#         alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
#         gradx1_s = (torch.sigmoid(gradx1) * 2) - 1
#         grady1_s = (torch.sigmoid(grady1) * 2) - 1
#         gradx2_s = (torch.sigmoid(gradx2 * alphax) * 2) - 1
#         grady2_s = (torch.sigmoid(grady2 * alphay) * 2) - 1
#         gradx_loss.append(torch.mean(torch.mul(torch.square(gradx1_s), torch.square(gradx2_s))))
#         grady_loss.append(torch.mean(torch.mul(torch.square(grady1_s), torch.square(grady2_s))))
#         img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
#         img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)
    
#     loss_gradxy = sum(gradx_loss)/level + sum(grady_loss)/level
#     return loss_gradxy

# img1 = torch.rand(3, 256, 256)
# img2 = torch.rand(3, 256, 256)
# loss = grad_separation_loss(img1, img2)
# print(loss)
# img = imageio.imread('/home/lhy/research/intrinsic-gs/datasets/intrinsic/50/replica/office_0/images/rgb_36.png')

# sf_img =get_sf_image(img)
# sf_torch = torch.from_numpy(sf_img)
# img = img / 255
# hf_img = get_hf_image_np(img)
# hf_torch = torch.from_numpy(hf_img)


# print(sf_torch.is_contiguous())
# print(hf_torch.is_contiguous())

# img = torch.rand(3, 480,640)
# print(img.is_contiguous())
# img = img.transpose(0,1)
# print(img.is_contiguous())
# # hf_img = get_hf_image(img)
# hf_torch = torch.from_numpy(hf_img)
# print(hf_img.is_contiguous())
# import torch

# # 创建一个张量并进行转置操作
# x = torch.randn(3, 4, 5)  # 原始张量
# y = x.permute(1, 2, 0)    # 经过 permute 操作后的新张量

# print("Original tensor:", x.is_contiguous())  # 检查原始张量是否连续
# print("Permuted tensor:", y.is_contiguous())  # 检查新张量是否连续

# import torch

# # 创建一个连续的 PyTorch 张量
# x = torch.randn(2, 3, 2)  # 形状为 [2, 3]

# # 检查原始 PyTorch 张量是否连续
# print("Is original tensor contiguous:", x.is_contiguous())
# x = x.transpose(0, 1)
# # 将 PyTorch 张量转换为 NumPy 数组
# numpy_arr = x.numpy()

# # 检查得到的 NumPy 数组是否连续
# print("Is NumPy array contiguous:", numpy_arr.flags['C_CONTIGUOUS'])

# from utils.loss_utils import get_separation_loss

# # img1 = torch.rand(3, 256, 256).cuda()
# # img2 = torch.rand(3, 256, 256).cuda()
# # loss = get_separation_loss(img1, img2)
# # print(loss)

# feat = torch.rand(4, 480, 640).cuda()
# reflect, shading  = feat.split([3, 1], dim=0)
# # print(reflect.is_contiguous())
# # print(shading.is_contiguous())

# loss = 0.0 


# loss = loss + get_separation_loss(reflect, shading)
# print(loss)

# import torch

# 创建两个连续的张量

# x1 = torch.randn(2, 3)  # 形状为 [2, 3]
# x2 = torch.randn(2, 3)  # 形状为 [2, 3]

# # 使用 torch.cat() 拼接张量
# y = torch.cat((x1, x2), dim=0)  # 沿着维度 0 进行拼接

# # 检查拼接后的张量是否连续
# print("Is concatenated tensor contiguous:", y.is_contiguous())



# import cv2

# # 读取图像
# image_bgr = cv2.imread('/home/lhy/data1/code/intrinsic-gs/datasets/intrinsic/replica/room_0/images/rgb_0.png')

# image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

# # y_channel = image_ycrcb[:,:,0]
# # image_ycrcb[:,:,0] = 0
# # 保存Y通道为单通道图像
# cv2.imwrite('y_channel_image.jpg', image_ycrcb[...,0])
# cv2.imwrite('y_channel_image_5.jpg', image_ycrcb[...,0] * 0.5)
# cv2.imwrite('cr_channel_image.jpg', image_ycrcb[:,:,1])
# cv2.imwrite('cb_channel_image.jpg', image_ycrcb[:,:,2])
# cv2.imwrite('cr_channel_imssage.jpg', image_ycrcb)

# 将图像转换为灰度图
# 将图像转换为灰度图
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 使用拉普拉斯算子进行边缘检测
# laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# # 将结果转换为无符号8位整数类型
# laplacian = cv2.convertScaleAbs(laplacian)
# # laplacian = np.where(laplacian > 25, 255, 0)
# # print(laplacian)
# # print(laplacian.max(), laplacian.min())
# # 显示结果
# cv2.imwrite('laplacian_ri.jpg', laplacian)
# lap  =torch.from_numpy(laplacian).float()

# new = torch.where(lap>0.1, True, False).repeat(3, 1, 1)
# print(new.shape)
# print(new.nonzero().shape)
# img = torch.ones(3, 480, 640)
# # print(img[new].shape)
# import torchvision.transforms.functional as F

# def rgb_to_lab(image):
#     # 将RGB图像转换为PIL图像
#     image_pil = F.to_pil_image(image)

#     # 将PIL图像转换为Lab色彩空间
#     image_lab = F.to_tensor(F.rgb_to_lab(image_pil))

#     return image_lab

# # 例子
# image_rgb = torch.rand(3, 256, 256)  # 生成一个随机的RGB图像，范围在0到1之间
# image_lab = rgb_to_lab(image_rgb)
# import cv2

# image = cv2.imread('/home/lhy/data1/code/intrinsic-gs/datasets/intrinsic/replica/office_3/images/rgb_348.png')
# cv2.imwrite('image.jpg', image)
# # 将图像转换为 LAB 颜色空间
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# print(lab_image[:, :, 0].max(), lab_image[:, :, 0].min())
# print(lab_image[:, :, 1].max(), lab_image[:, :, 1].min())
# print(lab_image[:, :, 2].max(), lab_image[:, :, 2].min())
# cv2.imwrite('LAB.jpg', lab_image)
# cv2.imwrite('L_channel_image.jpg', lab_image[:, :, 0])
# cv2.imwrite('A_channel_image.jpg', lab_image[:, :, 1])
# cv2.imwrite('B_channel_image.jpg', lab_image[:, :, 2])

# import os

# lab_path = ''
# weights = ''
# gbr_image = cv2.imread('/home/lhy/data1/code/intrinsic-gs/datasets/100/room_0/images/rgb_354.png')
# lab_image = cv2.cvtColor(gbr_image, cv2.COLOR_BGR2LAB)

# cv2.imwrite('lab_image', lab_image.astype('uint8'))



# laplacian = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# cv2.imwrite('laplacian.jpg', laplacian[:,:,1])
# print(laplacian[:,:,0].max(),laplacian[:,:,0].min())
# print(laplacian[:,:,1].max(),laplacian[:,:,1].min())
# print(laplacian[:,:,2].max(),laplacian[:,:,2].min())

# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
# gradient_direction = np.arctan2(sobel_y, sobel_x)

# cv2.imwrite('sobel_x.jpg', np.abs(sobel_x).astype('uint8'))
# cv2.imwrite('sobel_y.jpg', np.abs(sobel_y).astype('uint8'))
# cv2.imwrite('gradient_magnitude.jpg', gradient_magnitude.astype('uint8'))
# cv2.imwrite('gradient_direction.jpg', (gradient_direction / np.pi * 180).astype('uint8'))  # 将弧度转换为角度

# import torch
# import cv2

# def rgb2lab(rgb):
#     with torch.no_grad():
#         rgb = rgb.permute(1, 2, 0)
#         lab = cv2.cvtColor((rgb.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2LAB) / 255.0
#         lab = torch.from_numpy(lab).permute(2, 0, 1).contiguous().cuda()
#     return lab
    

# def cal_lab_weights(origin_image):
#     origin_lab = rgb2lab(origin_image)
#     f = torch.stack([origin_lab[0] * 0.3, origin_lab[1], origin_lab[2]])
#     return f.norm(dim=0)


# image = torch.rand(3, 100,300).cuda()

# weights = cal_lab_weights(image)
# print(weights.shape)

# def rgb2srgb(rgb):
#     srgb = torch.where(rgb > 0.0031308, torch.pow(1.055 * rgb, 1.0 / 2.4) - 0.055, rgb * 12.92)
#     return srgb
#     ret = torch.zeros_like(rgb)
#     idx0 = rgb <= 0.0031308
#     idx1 = rgb > 0.0031308
#     ret[idx0] = rgb[idx0] * 12.92
#     ret[idx1] = torch.pow(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
#     return ret


# def srgb2rgb(srgb):
#     rgb = torch.where(srgb > 0.04045, torch.pow((srgb + 0.055) / 1.055, 2.4), srgb / 12.92)
#     return rgb

# image = cv2.imread('/home/lhy/data1/code/intrinsic-gs/datasets/100/room_0/images/rgb_354.png')
# cv2.imwrite('origin.jpg', image)
# image = torch.from_numpy(image / 255).float().permute(2, 0, 1)

# srgb = srgb2rgb(image)

# srgb_image = (srgb.permute(1, 2, 0).numpy() * 255).astype('uint8')
# print(srgb_image.max(), srgb_image.min())
# cv2.imwrite('srgb.jpg', srgb_image)
# # import torch

# # 创建两个张量
# x1 = torch.tensor([[1., 2, 3]])
# x2 = torch.tensor([[4., 5, 6]])

# # 计算余弦相似度
# similarity = torch.nn.functional.cosine_similarity(x1, x2)
# print(similarity.item())  # 输出结果

import torch

# # 创建一个示例张量
# x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float)

# # 计算张量的第 50% 百分位数
# percentile_50 = torch.percentile(x, q=50)
# print("50th percentile:", percentile_50.item())

# # 计算张量的第 25% 百分位数
# percentile_25 = torch.percentile(x, q=25)
# print("25th percentile:", percentile_25.item())

# # 计算张量的第 75% 百分位数
# percentile_75 = torch.percentile(x, q=75)
# print("75th percentile:", percentile_75.item())

# # shading = torch.rand(3, 200, 300).cuda()
# # shading = adjust_image_for_display(shading, False).repeat(3, 1, 1)
# # print(shading.shape)

# # a = torch.arange(10)
# # b = torch.arange(15)

# def get_window(index, window_size, height, width):
#     row_index = index % height
#     col_index = index // height + 1
#     p = torch.arange(row_index, min(height, row_index + window_size))
#     q = torch.arange(col_index, min(width, col_index + window_size))
#     x, y = torch.meshgrid(p, q)
#     x = x.flatten()
#     y = y.flatten()
#     id = (y - 1) * height + x
#     id = id[1:]  # 去除中心像素的索引
#     win = torch.stack((index * torch.ones_like(id), id), dim=1)
#     return win

# image = torch.arange(15)[None].repeat(10,1).contiguous()

# window = get_window()


# def light_smooth_loss(light, normal):
    # 
    
    
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from kornia.geometry import depth_to_normals


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

def cal_laplacian(data):
    """
    data: [1, c, H, W]
    """
    # kernel = [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
    kernel = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(data.device)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    laplacian = F.conv2d(data, weight, padding='same')
    return laplacian

def cal_gradient(data, p=1):
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
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    grad = torch.cat([grad_x, grad_y], dim=-3)
    gradient = torch.norm(grad, p=p, dim=-3, keepdim=True)
    return gradient

def light_smooth_loss(light, normal, grad_threshold=0.05):
    normal_grad = cal_gradient(torch.abs(normal).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    flatten_mask = torch.where(normal_grad < grad_threshold, True, False)
    light_mag = torch.linalg.norm(light, ord=2, dim=0, keepdim=True)
    mag_grad = cal_gradient(light_mag.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    mag_grad_loss = mag_grad[flatten_mask].mean()
    
    light_dir = light / light_mag
    dir_grad = cal_gradient(torch.abs(light_dir).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
    dir_smooth_loss = dir_grad.mean()
    
    return mag_grad_loss, dir_smooth_loss

def adjust_image_for_display(image, trans2srgb, rescale=True):
    MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837

    vis = image.clone()
    if rescale:
        s = np.percentile(image.cpu().numpy(), 99.9).item()
        if s > MAX_SRGB:
            vis = vis / s * MAX_SRGB

    vis = torch.clamp(vis, min=0)
    # vis[vis < 0] = 0
    if trans2srgb:
        vis[vis > MAX_SRGB] = MAX_SRGB
        vis = rgb_to_srgb(vis)
    return vis

# source_path = '/home/lhy/data1/code/intrinsic-gs/datasets/100/room_0'

# scene_info = readColmapSceneInfo(source_path, 'images', True)

from skimage.segmentation import felzenszwalb
from skimage import io, img_as_ubyte
seq = 408

bgr_image = cv2.imread(f'/home/lhy/data1/code/intrinsic-gs/datasets/100/room_0/images/rgb_{seq}.png')
# normal_image = imageio.imread('/home/lhy/data1/code/intrinsic-gs/output/replica/room_0/3dgs/eval/normal/rgb_63.png')

cv2.imwrite('test_image/rgb_image.jpg', bgr_image)
# imageio.imwrite('test_image/normal.jpg', normal_image)
image = io.imread(f'/home/lhy/data1/code/intrinsic-gs/datasets/100/room_0/images/rgb_{seq}.png')
segments_fz = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
io.imsave('test_image/fz.png', img_as_ubyte(segments_fz))

rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
lap_image = cv2.Laplacian(gray_image, cv2.CV_64F)
cv2.imwrite('test_image/lap_image.jpg', lap_image)

rgb = torch.from_numpy(rgb_image/255).float().permute(2, 0, 1)
lab = torch.from_numpy(lab_image/255).float().permute(2, 0, 1)
lap = torch.abs(torch.from_numpy(lap_image[..., None]/255).float().permute(2, 0, 1))
# normal = torch.from_numpy(normal_image/255 * 2 - 1).float().permute(2, 0, 1)

rgb_grad = cal_gradient(rgb.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)

lab_f = torch.stack([lab[0]*0.3, lab[1], lab[2]], dim=0)
lab_norm = torch.norm(lab_f,p=1, dim=0, keepdim=True)
lab_grad = cal_gradient((lab_norm**1).mean(0, keepdim=True).unsqueeze(0)).squeeze(0)

# normal_grad = cal_gradient(normal.abs().mean(0, keepdim=True).unsqueeze(0)).squeeze(0)

scale = 0.8
# lab_normal_grad = torch.where(lab_grad > scale * normal_grad , lab_grad, scale * normal_grad)
lab_rgb_grad = torch.where(lab_grad > scale * rgb_grad , lab_grad, scale * rgb_grad)

print(f'lab  grad: max{lab_grad.max()}--min{lab_grad.min()}')  # 7.823 0
print(f'rgb  grad: max{rgb_grad.max()}--min{rgb_grad.min()}')  # 6,    0
# print(f'lap  grad: max{lap.max()}--min{lap.min()}')  # 7.823 0
# print(f'unit grad: max{unit_grad.max()}--min{unit_grad.min()}')

cv2.imwrite('test_image/rgb_grad.jpg', (adjust_image_for_display(rgb_grad, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
cv2.imwrite('test_image/lab_grad.jpg', (adjust_image_for_display(lab_grad, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
# cv2.imwrite('test_image/lab_normal_grad.jpg', (adjust_image_for_display(lab_normal_grad, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
cv2.imwrite('test_image/lab_rgb_grad.jpg', (adjust_image_for_display(lab_rgb_grad, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
# cv2.imwrite('test_image/normal_grad.jpg', (adjust_image_for_display(normal_grad, True).permute(1, 2, 0).numpy()*255).astype('uint8'))

# lab_weight = 2 * torch.exp(-1 * lab_grad)
# lab_normal_weight = 2 * torch.exp(-1 * lab_normal_grad)
# lab_rgb_weight = 2 * torch.exp(-1 * lab_rgb_grad)

a = 40
b = 6
# 1 / (1 + exp(ax+b))
lab_weight = 1 / (1 + torch.exp(a * lab_grad - b))
# lab_normal_weight = 1 / (1 + torch.exp(a * lab_normal_grad - b))
lab_rgb_weight = 1 / (1 + torch.exp(a * lab_rgb_grad - b))



print(f'lab_weight max={lab_weight.max()}---,min={lab_weight.min()}')
# print(f'lab_normal_weight max={lab_normal_weight.max()}---,min={lab_normal_weight.min()}')
print(f'lab_rgb_weight max={lab_rgb_weight.max()}---,min={lab_rgb_weight.min()}')

cv2.imwrite('test_image/lab_weight.jpg', (adjust_image_for_display(lab_weight, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
# cv2.imwrite('test_image/lab_normal_weight.jpg', (adjust_image_for_display(lab_normal_weight, True).permute(1, 2, 0).numpy()*255).astype('uint8'))
cv2.imwrite('test_image/lab_rgb_weight.jpg', (adjust_image_for_display(lab_rgb_weight, True).permute(1, 2, 0).numpy()*255).astype('uint8'))

gt_image= rgb

value_img = torch.max(gt_image, dim=0, keepdim=True)[0]
shallow_enhance = gt_image
shallow_enhance = 1 - (1 - shallow_enhance) * (1 - shallow_enhance)

specular_enhance = gt_image
specular_enhance = specular_enhance * specular_enhance
k = 5
specular_weight = 1 / (1 + torch.exp(-k * (value_img - 0.5)))
target_img = (specular_weight * specular_enhance + (1 - specular_weight) * shallow_enhance)
# cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
cv2.imwrite('test_image/target_grad.jpg', cv2.cvtColor((adjust_image_for_display(target_img, True).permute(1, 2, 0).numpy()*255).astype('uint8'), cv2.COLOR_RGB2BGR))

# unit_weight = adjust_image_for_display(unit_weight, False)
# unit_weight_image = (unit_weight.permute(1, 2, 0).numpy()*255)
# # print(f'max:{weight.max()}, min:{weight.min()}')
# weight = adjust_image_for_display(lab_weight, False)
# # print(f'max:{weight.max()}, min:{weight.min()}')
# weight_image = (weight.permute(1, 2, 0).numpy()*255)
    
# weight_image = weight_image.astype('uint8')
# unit_weight_image = unit_weight_image.astype('uint8')
# rgb_grad = (rgb_grad.permute(1, 2, 0).numpy()*255).astype('uint8')
# print(f'max:{weight_image.max()}, min:{weight_image.min()}')
# cv2.imwrite('test_image/rgb_grad.jpg', rgb_grad)
# cv2.imwrite(f'test_image/weight_grad2.jpg', weight_image)
# cv2.imwrite(f'test_image/unit_weight_grad2.jpg', unit_weight_image)



# # 读取深度图像
# normal_image = cv2.imread('/home/lhy/data1/code/intrinsic-gs/output/replica/room_1/eval/normal/rgb_770.png')
# normal_image = normal_image / 255.0
# normal_image = normal_image * 2 -1
# # print(normal_image.max(), normal_image.min())
# normal = torch.from_numpy(normal_image).float().permute(2, 0, 1)

# light = torch.rand_like(normal)

# mag_grad_loss, dir_smooth_loss = light_smooth_loss(light, normal)
# print(mag_grad_loss, dir_smooth_loss)

# # print(normal_image.dtype)

# normal_lap = cal_gradient(torch.abs(normal).mean(0, keepdim=True).unsqueeze(0)).squeeze()
# # # print(np.count_nonzero(normal_lap > 0.01)/np.count_nonzero(normal_lap))
# # # print(np.percentile(normal_lap, 0.99))
# # # print(normal_lap.max())
# # # normal_lap = (normal_lap + 1 )/2
# print(normal_lap.max(), normal_lap.min())
# # # # nor = torch.abs(normal_lap).mean(0)
# print(np.count_nonzero(normal_lap > 0.05)/np.count_nonzero(normal_lap))
# normal_lap = (normal_lap.abs().numpy() * 255)
# print(normal_lap.max(), normal_lap.min())

# normal_lap = normal_lap.astype('uint8')
# print(normal_lap.max(), normal_lap.min())
# # # print(normal_lap.max(), normal_lap.min())
# # # # print(normal_lap.max(), normal_lap.min())
# cv2.imwrite('3nor.jpg', normal_lap)


# print(normal_image.dtype)
# 计算法线图
# gradient_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
# normal_map = np.dstack((-gradient_x, -gradient_y, np.ones_like(depth_image)))
# normal_map = normal_map / np.linalg.norm(normal_map, axis=2, keepdims=True)
# cv2.imwrite('normal.jpg', (normal_image - 1)/2)
# # 计算法线图的梯度
# normal_gradient_x = cv2.Sobel(normal_image[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
# normal_gradient_y = cv2.Sobel(normal_image[:,:,1], cv2.CV_64F, 0, 1, ksize=3)
# normal_gradient_magnitude = np.sqrt(normal_gradient_x**2 + normal_gradient_y**2)
# cv2.imwrite('normal_grad.jpg', normal_gradient_magnitude)


# normal_map_gray = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)

# # 使用拉普拉斯算子计算法线图的梯度
# laplacian_gradient = cv2.Laplacian(normal_map_gray, cv2.CV_64F)
# cv2.imwrite('lap.jpg', laplacian_gradient)

# 显示结果
# cv2.imshow('Depth Image', depth_image)
# cv2.imshow('Normal Map', normal_map)
# cv2.imshow('Normal Gradient Magnitude', (255 * normal_gradient_magnitude / np.max(normal_gradient_magnitude)).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = torch.arange(3* 100*100).reshape(3, 100, 100)
# image_np = image.permute(1, 2, 0)