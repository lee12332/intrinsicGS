
import json
import numpy as np
import torch
import math
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_sh_coef
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera
from utils.loss_utils import cal_gradient, cal_gradient_xy, get_smooth_weight, get_sparsity_weight, ssim, get_depth_loss
from utils.image_utils import cal_local_weights, get_chromaticity_image, get_hf_image, lab2flab, psnr, rgb2ycbcr
import torch.nn.functional as F
from kornia.geometry import depth_to_normals
from utils.image_utils import rgb2lab, rgb2lab_np


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()#默认张量的梯度会在算完后置为0，retain_grad()保留某个张量的梯度，用于后续计算
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics#内参矩阵，注意区分内参矩阵的intrinsic和本征属性的intrinsic

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:#预设的颜色
        if pipe.compute_SHs_python:#这个if线根本就没有进入而是进入了else，因为get_shs函数是没有定义的
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,#相机到每个高斯点的方向向量，并归一化
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    reflectance = pc.get_reflectance # [N, 3]
    shading = pc.get_shading # [N, 1]
    residual = pc.get_residual # [N, 16, 3] 为什么残差是这个维度？
    offset = pc.get_offset # [N, 3]
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    # print("shape of viewdirs")
    # print(viewdirs.shape)
    # print("shape of camera_center")
    # print(viewpoint_camera.camera_center.shape)
    # print("shape of 3d")
    # print(means3D.shape)
    
    
    if pipe.compute_intrinsic_composition:
        intrinsic_color = intrinsic_composition(reflectance, shading, offset, residual, viewdirs, use_residual=pipe.use_residual)#这个函数就是c=(reflect+delta）*shading+residual
        
        feature_list = [intrinsic_color, reflectance, shading, offset]
        ch_list = [3, 3, 1, 3]
        
            
        if pipe.use_residual: 
            deg = int(np.sqrt(residual.shape[1]) - 1)
            residual_shs_view = residual.transpose(1, 2).view(-1, 3, residual.shape[1]) # [N, 3, 16]
            residual_view = eval_sh(deg, residual_shs_view, viewdirs) # [N, 3]
            residual_view = torch.clamp_min(residual_view + 0.5, 0.0)
            feature_list.append(residual_view)#feature_list = [intrinsic_color, reflectance, shading, offset, residual_view]，python list的元素类型可以不同
            ch_list.append(residual_view.shape[-1])#这个list存的是feature list的维度
            
        features = torch.cat(feature_list, dim=-1)#[N, 3 + 3 + 1 + 3 + 3]
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        #TODO 一大问题是本征属性在cuda代码的哪里实现的，在r3dg_rasterization的forward.cu的renderCUDA中？
        #TODO rendered_image和rendered_list[0]（rendered_intrinsic）分别是什么，rendered_image是否参与了优化，如果是能否不优化这个来节约时间
        (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
        rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            features=features,
        )
        # print(rendered_opacity.shape,rendered_opacity.mean())

        feature_dict = {}
        rendered_list = rendered_feature.split(ch_list, dim=0)
        rendered_intrinsic = rendered_list[0]#这个intrinsic本征方程合成的图像吗？
        rendered_reflectance = rendered_list[1]
        rendered_shading = rendered_list[2]
        rendered_offset = rendered_list[3]
        # rendered_normal = rendered_list[3]
        feature_dict['intrinsic'] = rendered_intrinsic
        feature_dict['reflectance'] = rendered_reflectance
        feature_dict['shading'] = rendered_shading
        feature_dict['offset'] = rendered_offset
        # feature_dict['normal'] = rendered_normal    
        

        if pipe.use_residual:
            rendered_residual = rendered_list[-1]
            feature_dict["residual"] = rendered_residual


        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        results = {"render": rendered_image,
                #    "pseudo_normal": rendered_pseudo_normal,
                   "surface_xyz": rendered_surface_xyz,
                   "opacity": rendered_opacity,
                   "depth": rendered_depth,
                   "viewspace_points": screenspace_points,
                   "visibility_filter": radii > 0,
                   "radii": radii,
                   "num_rendered": num_rendered,
                   "num_contrib": num_contrib
                }

        results.update(feature_dict)#把字典 dict2 的键/值对更新到 dict 里。

        return results

        
        

calls=0
def calculate_loss(viewpoint_camera, pc, results, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0]
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    # rendered_normal = results["normal"]
    rendered_intrinsic = results["intrinsic"]
    rendered_reflectance = results["reflectance"]
    rendered_shading = results["shading"]
    rendered_opacity = results['opacity']
    rendered_offset = results['offset']
    
    
    if opt.lambda_residual > 0:
        rendered_residual = results['residual']


    
    gt_image = viewpoint_camera.original_image.cuda()

    loss_dict={}
    loss = 0.0
    global calls
    loss_dict.update({"calls":calls})
    if opt.lambda_vanilla > 0:#这个参数似乎一直是0，衡量render_image和gt_image的损失
    
        Ll1 = F.l1_loss(rendered_image, gt_image)
        ssim_val = ssim(rendered_image, gt_image)
        
        tb_dict["l1"] = Ll1.item()
        tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
        tb_dict["ssim"] = ssim_val.item()

        loss = loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    
    if opt.lambda_intrinsic > 0:#衡量rendered_intrinsic和gt_image的损失
        Ll1_intrinsic = F.l1_loss(rendered_intrinsic, gt_image)
        ssim_val_intrinsic = ssim(rendered_intrinsic, gt_image)
        
        tb_dict["l1_intrinsic"] = Ll1_intrinsic.item()
        tb_dict["psnr_intrinsic"] = psnr(rendered_intrinsic, gt_image).mean().item()
        tb_dict["ssim_intrinsic"] = ssim_val_intrinsic.item()
        
        loss_intrinsic = (1.0 - opt.lambda_dssim) * Ll1_intrinsic + opt.lambda_dssim * (1.0 - ssim_val_intrinsic)
        loss = loss + opt.lambda_intrinsic * loss_intrinsic
        loss_dict.update({"intrinsic_loss":opt.lambda_intrinsic * loss_intrinsic.item()})


    #一般colmap估出来的位置是相对的，真实数据集（传感器）估出来的是真实值，尺度不一致，为什么合成数据集不存在这个问题？
    if opt.lambda_depth > 0:#合成数据集，尺度一致，衡量gt_depth和rendered_depth的L1损失
        gt_depth = viewpoint_camera.depth.cuda() # gt 深度，与colmap深度尺度一致
        # image_mask = viewpoint_camera.image_mask.cuda().bool()
        # depth_mask = gt_depth > 0
        # sur_mask = torch.logical_xor(image_mask, depth_mask)
        
        # loss_depth = get_depth_loss(rendered_depth, gt_depth, ~sur_mask)
        loss_depth = F.l1_loss(gt_depth, rendered_depth)
        
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_scale_depth > 0: #真实数据集，尺度不一致，使用尺寸不变损失Depth map prediction from a single image using a multi-scale deep network gt_depth和rendered_depth，mask调用get_depth_loss计算损失，不清楚image_mask是什么
        gt_depth = viewpoint_camera.depth.cuda() # gt 深度，与colmap深度尺度不一致
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)
        
        loss_depth = get_depth_loss(rendered_depth, gt_depth, ~sur_mask)
        # loss_depth = F.l1_loss(gt_depth, rendered_depth)
        
        loss = loss + opt.lambda_depth * loss_depth


    """intrinsic part"""
    #gt_image.shape=[3,H,W]
    gt_depth = viewpoint_camera.depth.cuda()#[1,H,W]
    gt_normal = viewpoint_camera.normal.cuda()#[3,H,W]
    gt_lab = viewpoint_camera.lab.cuda()#[3,H,W]
    depth_mask = (gt_depth > 0).float()
    ## depth to normal, if there is a gt depth but not a MVS normal map 在没有mvs法线图的情况下更新viewpoint_camera中的法线
    if torch.allclose(gt_normal, torch.zeros_like(gt_normal)):#allclose是比较两个张量在容忍限度内是否相等的函数，法线跟零向量相似，用深度求法线并更新
        from kornia.geometry import depth_to_normals
        normal_pseudo_cam = -depth_to_normals(gt_depth[None], viewpoint_camera.intrinsics[None])[0]#kornia库，使用深度的梯度估计法线，法线的存储方式是(B,3,H,W,)三个rgb通道存储的该像素法线的方向向量，负号调整方向
        # trans=viewpoint_camera.world_view_transform
        # cam_intrinsics=viewpoint_camera.intrinsics
        c2w = viewpoint_camera.world_view_transform.T.inverse()
        R = c2w[:3, :3]
        _, H, W = normal_pseudo_cam.shape
        gt_normal = (R @ normal_pseudo_cam.reshape(3, -1)).reshape(3, H, W)#@是矩阵乘法
        viewpoint_camera.normal = gt_normal
    
    if opt.lambda_reflectance_sparsity > 0:#反射率分段稀疏
        sparsity_weight= get_sparsity_weight(gt_image, gt_normal, gt_lab)
        reflectance_grad = cal_gradient(rendered_reflectance.mean(0, keepdim=True).unsqueeze(0)).squeeze(0)#unsqueeze()，在指定位置添加一个形状为1的维度，一般用来改变形状以满足某些惭怍的形状要求，squeeze(0)与之相反
        sparsity_loss = (sparsity_weight * reflectance_grad).mean()#直接调用mean()就是求所有元素的平均值，mean(dim,keepdim)就是求某个维度的均值，改维度形状会变为1，用keepdim决定包不包六该维度
        loss = loss + sparsity_loss * opt.lambda_reflectance_sparsity
        loss_dict.update({"sparsity_loss":opt.lambda_reflectance_sparsity*sparsity_loss.item()})
    
    if opt.lambda_shading_smooth > 0:#照明平滑
        smooth_weight = get_smooth_weight(gt_depth, gt_normal, gt_lab)
        shading_grad = cal_gradient(rendered_shading.mean(0, keepdim=True).unsqueeze(0), p=2).squeeze(0) **2
        # shading_smooth_loss = shading_grad.mean()
        normal_grad = cal_gradient(gt_normal.abs().mean(0, keepdim=True).unsqueeze(0)).squeeze(0)
        smooth_mask = torch.where(normal_grad < 0.1, True, False)#torch.where(condition, x, y)根据condition选择x或者y
        shading_smooth_loss = (smooth_weight * shading_grad)[smooth_mask].mean()
        # shading_smooth_loss = (smooth_weight * shading_grad).mean()
        loss = loss + opt.lambda_shading_smooth * shading_smooth_loss
        loss_dict.update({"smooth_loss":opt.lambda_shading_smooth * shading_smooth_loss.item()})
    
    if opt.lambda_prior_chromaticity > 0:#反射率色度损失
        prior_chrom = lab2flab(rgb2lab(gt_image.permute(1, 2, 0)))
        reflect = torch.clamp(rendered_reflectance + rendered_offset, 0, 1)
        reflect_chrom = lab2flab(rgb2lab(reflect.permute(1, 2, 0)))
        loss_prior_chromaticity = ((prior_chrom - reflect_chrom) ** 2).mean()
        loss = loss + loss_prior_chromaticity * opt.lambda_prior_chromaticity
        loss_dict.update({"chrom_loss":opt.lambda_prior_chromaticity*loss_prior_chromaticity.item() })

    if opt.lambda_residual > 0:
        loss_residual = (torch.norm(rendered_residual, dim=0)**2).mean()#torch.norm，对某个维度求p范数，默认为2，为什么这里残差不用梯度求，而是直接用L2
        loss = loss + opt.lambda_residual * loss_residual 
        loss_dict.update({"residual_loss":opt.lambda_residual * loss_residual.item() })

    if opt.lambda_offset > 0:
        loss_offset = (torch.norm(rendered_offset, dim=0)**2).mean()
        loss = loss + opt.lambda_offset * loss_offset  
        loss_dict.update({"offset_loss":opt.lambda_offset * loss_offset.item()  })  


    loss_dict.update({"total_loss":loss.item()  })
    tb_dict["loss"] = loss.item()


    if calls%500==0:
        data_str = json.dumps(loss_dict, indent=4)
        with open("loss_dict.txt", "a") as file:
            file.write(data_str + "\n")  # 每次写入后换行
    calls+=1

    return loss, tb_dict


def intrinsic_composition(reflectance, shading, offset, residual, view_dirs, use_residual):#c=(reflect+delta）*shading+residual
    # intrinsic_rgb = reflectance * shading
    intrinsic_rgb = torch.clamp(reflectance + offset, 0, 1) * shading#(reflect+delta）*shading, torch.clamp限制幅度函数，
    if use_residual:
        deg = int(np.sqrt(residual.shape[1]) - 1)#求残差的球谐度数deg
        residual_shs_view = residual.transpose(1, 2).view(-1, 3, residual.shape[1]) # [N, 3, 16] .view()相当于resize，-1表示这里的维度长度取决于其他维度数
        residual_view = eval_sh(deg, residual_shs_view, view_dirs) # [N, 3]
        residual_view = torch.clamp_min(residual_view + 0.5, 0.0)
        intrinsic_rgb = intrinsic_rgb + residual_view#c=(reflect+delta）*shading+residual
    
    return intrinsic_rgb

    
def render_intrinsic(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                    scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                    is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results