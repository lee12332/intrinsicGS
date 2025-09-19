import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        #激活函数确保参数在合理范围内，确保在优化空间的数据在激活后保持其本身的性质，如非负，模长
        # self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize#确保四元数长为1
        
        if self.use_intrinsic:
            self.reflectance_activation = torch.sigmoid
            self.intensity_activation = torch.sigmoid
            self.light_activation = torch.relu

    def __init__(self, sh_degree : int, use_intrinsic=False, geo_enhence=False):
        self.use_intrinsic = use_intrinsic
        self.geo_enhence = geo_enhence
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  # 阶数=2
        self._xyz = torch.empty(0)
        # self._normal = torch.empty(0)  # normal
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        # self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        if self.use_intrinsic:
            self._reflectance = torch.empty(0)
            self._intensity = torch.empty(0)
            self._residual_dc = torch.empty(0)
            self._residual_rest = torch.empty(0)
            self._light = torch.empty(0)
            self._offset = torch.empty(0)
            
        self.setup_functions()

    def capture(self):#capture和restore获取整个高斯场景的属性数据
        captured_list = [
            self.active_sh_degree,
            self._xyz,
            # self._normal,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            # self.normal_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ]
        if self.use_intrinsic:
            captured_list += [self._reflectance, 
                              self._intensity, 
                              self._residual_dc, 
                              self._residual_rest,
                              self._light,
                              self._offset]
            
        return tuple(captured_list)
    
    def restore(self, model_args, training_args, is_training=False, restore_optimizer=True):

        (self.active_sh_degree, 
        self._xyz, 
        # self._normal,
        self._features_dc,
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        # normal_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args[:12]
        
        if len(model_args) > 12 and self.use_intrinsic:
            (self._reflectance,
            self._intensity,
            self._residual_dc,
            self._residual_rest,
            self._light,
            self._offset) = model_args[12:]
        
        if is_training:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            # self.normal_gradient_accum = normal_gradient_accum
            self.denom = denom
            if restore_optimizer:
                # TODO automatically match the opt_dict
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    # @property
    # def get_normal(self):
    #     return self.normal_activation(self._normal)
    
    # @property
    # def get_light_intensity(self):
    #     return torch.norm(self._light, dim=-1, keepdim=True)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_reflectance(self):
        return self.reflectance_activation(self._reflectance)

    @property
    def get_intensity(self):
        return self.intensity_activation(self._intensity)
    
    @property
    def get_light(self):
        return self._light
    
    @property
    def get_shading(self):
        return self.intensity_activation(self._intensity)

    @property
    def get_residual(self):
        residual_dc = self._residual_dc
        residual_rest = self._residual_rest
        return torch.cat((residual_dc, residual_rest), dim=1)

    @property
    def get_offset(self):
        return self._offset
    
    #  train.py 里每过1k次iter增加一阶，最高不超过3阶
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def attribute_names(self):
        # attribute_names = ['xyz', 'normal', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']
        attribute_names = ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']
        
        if self.use_intrinsic:
            attribute_names += ['reflectance', 'intensity','residual_dc', 'residual_rest', 'light', 'offset']
            
        return attribute_names

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)

        (self.active_sh_degree,
         self._xyz,
        #  self._normal,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
        #  normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:12]

        self.xyz_gradient_accum = xyz_gradient_accum
        # self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        if self.use_intrinsic:
            if len(model_args) > 12: # 有intrinsic
                (self._reflectance,
                self._intensity,
                self._residual_dc,
                self._residual_rest,
                self._light,
                self._offset) = model_args[12:]
            else: # 无intrinsic
                self._reflectance= nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True)) # [N_pts, 3]
                self._intensity = nn.Parameter(torch.ones_like(self._xyz[..., :1]).requires_grad_(True)) # [N_pts, 1]
                
                residual = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
                self._residual_dc = nn.Parameter(residual[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))   #[N_pts,  1, 3]
                self._residual_rest = nn.Parameter(residual[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  #[N_pts, 15, 3]
                
                self._light= nn.Parameter(torch.rand_like(self._xyz).requires_grad_(True)) # [N_pts, 3]
                self._offset = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True)) # [N_pts, 3]
                
        if restore_optimizer:
            # TODO automatically match the opt_dict
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter    
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # [N_pts, 3]
        # fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # [N_pts, 3]
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_intrinsic:
            reflectance = torch.zeros_like(fused_point_cloud)
            intensity = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            residual = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            light = torch.ones_like(fused_point_cloud)
            offset = torch.zeros_like(fused_point_cloud)
                                         
            self._reflectance = nn.Parameter(reflectance.requires_grad_(True))
            self._intensity = nn.Parameter(intensity.requires_grad_(True))
            self._residual_dc = nn.Parameter(residual[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._residual_rest = nn.Parameter(residual[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._light = nn.Parameter(light.requires_grad_(True))
            self._offset = nn.Parameter(offset.requires_grad_(True))
  
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.use_intrinsic:
            l += [
                {'params': [self._reflectance], 'lr': training_args.reflectance_lr, "name": "reflectance"},
                {'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"},
                {'params': [self._residual_dc], 'lr': training_args.residual_lr, "name": "residual_dc"},
                {'params': [self._residual_rest], 'lr': training_args.residual_rest_lr, "name": "residual_rest"},
                {'params': [self._light], 'lr': training_args.light_lr, "name": "light"},
                {'params': [self._offset], 'lr': training_args.offset_lr, "name": "offset"},]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
                
        if self.use_intrinsic:
            for i in range(self._reflectance.shape[1]): # 
                l.append('reflectance_{}'.format(i))
            l.append('intensity')
            for i in range(self._residual_dc.shape[1]*self._residual_dc.shape[2]):
                l.append('residual_dc_{}'.format(i))
            for i in range(self._residual_rest.shape[1]*self._residual_rest.shape[2]):
                l.append('residual_rest_{}'.format(i))
            for i in range(self._light.shape[1]): # 
                l.append('light_{}'.format(i))
            for i in range(self._offset.shape[1]): # 
                l.append('offset_{}'.format(i))
                
        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # normal = self._normal.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # attributes_list = [xyz, normal, f_dc, f_rest, opacities, scale, rotation]
        attributes_list = [xyz,normals, f_dc, f_rest, opacities, scale, rotation]
        
        if self.use_intrinsic:
            reflectance = self._reflectance.detach().cpu().numpy()
            intensity = self._intensity.detach().cpu().numpy()
            residual_dc = self._residual_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            residual_rest = self._residual_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            light = self._light.detach().cpu().numpy()
            offset = self._offset.detach().cpu().numpy()
            attributes_list += [reflectance, intensity, residual_dc, residual_rest, light, offset]
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self,  path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
        #                    np.asarray(plydata.elements[0]["ny"]),
        #                    np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        
        if self.use_intrinsic:
            reflectance_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("reflectance")]
            reflectance_names = sorted(reflectance_names, key = lambda x: int(x.split('_')[-1]))      
            assert len(reflectance_names)==3, 'reflectance长度为3，不是{}'.format(len(reflectance_names))
            reflectance = np.zeros((xyz.shape[0], len(reflectance_names)))
            for idx, attr_name in enumerate(reflectance_names):
                reflectance[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            intensity = np.asarray(plydata.elements[0]["intensity"])[..., np.newaxis]
                 
            residual_dc = np.zeros((xyz.shape[0], 3, 1))
            residual_dc[:, 0, 0] = np.asarray(plydata.elements[0]["residual_dc_0"])
            residual_dc[:, 1, 0] = np.asarray(plydata.elements[0]["residual_dc_1"])
            residual_dc[:, 2, 0] = np.asarray(plydata.elements[0]["residual_dc_2"])
            
            extra_res_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("residual_rest_")]
            extra_res_names = sorted(extra_res_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_res_names)==3*(self.max_sh_degree + 1) ** 2 - 3 , 'residual_rest长度为{}, 不是{}'.format(len(extra_res_names), 3*(self.max_sh_degree + 1) ** 2 - 3)
            residual_extra = np.zeros((xyz.shape[0], len(extra_res_names))), 
            for idx, attr_name in enumerate(extra_res_names):
                residual_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            residual_extra = residual_extra.reshape((residual_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            
            light_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("light_")]
            light_names = sorted(light_names, key = lambda x: int(x.split('_')[-1]))      
            assert len(light_names)==3, 'light长度为3，不是{}'.format(len(light_names))
            light = np.zeros((xyz.shape[0], len(light_names)))
            for idx, attr_name in enumerate(light_names):
                light[:, idx] = np.asarray(plydata.elements[0][attr_name])
                
                
            offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("offset")]
            offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))      
            assert len(offset_names)==3, 'offset长度为3，不是{}'.format(len(offset_names))
            offset = np.zeros((xyz.shape[0], len(offset_names)))
            for idx, attr_name in enumerate(offset_names):
                offset[:, idx] = np.asarray(plydata.elements[0][attr_name])
                 
            self._reflectance = nn.Parameter(torch.tensor(reflectance, dtype=torch.float, device="cuda").requires_grad_(True))
            self._intensity = nn.Parameter(torch.tensor(intensity, dtype=torch.float, device="cuda").requires_grad_(True))
            self._residual_dc = nn.Parameter(torch.tensor(residual_dc, dtype=torch.float, device="cuda").requires_grad_(True))
            self._residual_rest = nn.Parameter(torch.tensor(residual_extra, dtype=torch.float, device="cuda").requires_grad_(True))
            self._light = nn.parameter(torch.tensor(light, dtype=torch.float, device="cuda").requires_grad_(True))
            self._offset = nn.parameter(torch.tensor(offset, dtype=torch.float, device="cuda").requires_grad_(True))
            
            
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask): # mask 删除的点
        valid_points_mask = ~mask # 保留的点
        optimizable_tensors = self._prune_optimizer(valid_points_mask) # 筛选点云

        self._xyz = optimizable_tensors["xyz"]
        # self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_intrinsic:
            self._reflectance = optimizable_tensors["reflectance"]
            self._intensity = optimizable_tensors["intensity"]
            self._residual_dc = optimizable_tensors["residual_dc"]
            self._residual_rest = optimizable_tensors["residual_rest"]
            self._light = optimizable_tensors["light"]
            self._offset = optimizable_tensors["offset"]
            

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, 
                              new_xyz, 
                              # new_normal, 
                              new_features_dc, 
                              new_features_rest, 
                              new_opacity, 
                              new_scaling, 
                              new_rotation,
                              new_reflectance=None,
                              new_intensity=None,
                              new_residual_dc=None,
                              new_residual_rest=None,
                              new_light=None,
                              new_offset=None):
        d = {"xyz": new_xyz,
            #  'normal': new_normal,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacity,
             "scaling" : new_scaling,
             "rotation" : new_rotation}
        
        if self.use_intrinsic:
            assert (new_reflectance!=None and new_intensity!=None and new_residual_dc!=None and new_residual_rest!=None and new_light!=None and new_offset!=None)
            d['reflectance'] = new_reflectance
            d['intensity'] = new_intensity
            d['residual_dc'] = new_residual_dc
            d['residual_rest'] = new_residual_rest
            d['light'] = new_light
            d['offset'] = new_offset

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        # self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.use_intrinsic:
            self._reflectance = optimizable_tensors["reflectance"]
            self._intensity = optimizable_tensors["intensity"]
            self._residual_dc = optimizable_tensors["residual_dc"]
            self._residual_rest = optimizable_tensors["residual_rest"]
            self._light = optimizable_tensors["light"]
            self._offset = optimizable_tensors["offset"]


    # def densify_and_split(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold, N=2): # N的意思是每个点复制N次
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda") # 用0填充
        # padded_grad[:grads.shape[0]] = grads.squeeze() # 将grads的值填充到padded_grad中，形状为(n_init_points)
        # padded_grad_normal = torch.zeros((n_init_points), device="cuda") # 用0填充
        # padded_grad_normal[:grads_normal.shape[0]] = grads_normal.squeeze() # 将grads_normal的值填充到padded_grad_normal中，形状为(n_init_points)
        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # 选出梯度大于阈值的点
        # selected_pts_mask_normal = torch.where(padded_grad_normal >= grad_normal_threshold, True, False) # 选出法向梯度大于阈值的点
        # # print("densify_and_split_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])

        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal) # 选出梯度大于阈值或法向梯度大于阈值的点
        # selected_pts_mask = torch.logical_and(
        #     selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent) # 选出梯度大于阈值或法向梯度大于阈值的点，且其尺度大于一定比例的场景尺度
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # args = [new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        args = [new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        
        if self.use_intrinsic:
            new_reflectance = self._reflectance[selected_pts_mask].repeat(N, 1)
            new_intensity = self._intensity[selected_pts_mask].repeat(N, 1)
            new_residual_dc = self._residual_dc[selected_pts_mask].repeat(N, 1, 1)
            new_residual_rest = self._residual_rest[selected_pts_mask].repeat(N, 1, 1)
            new_light = self._light[selected_pts_mask].repeat(N, 1)
            new_offset = self._offset[selected_pts_mask].repeat(N, 1)
            args += [new_reflectance, new_intensity, new_residual_dc, new_residual_rest, new_light, new_offset]
            
        self.densification_postfix(*args)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold):
    def densify_and_clone(self, grads, grad_threshold, scene_extent):

        # Extract points that satisfy the gradient condition
        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask_normal = torch.where(torch.norm(grads_normal, dim=-1) >= grad_normal_threshold, True, False)
        # # print("densify_and_clone_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling,
        #                                                 dim=1).values <= self.percent_dense * scene_extent)
        
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        
        # 通过简单地创建相同大小的副本并将其沿位置梯度的方向移动来克隆高斯
        new_xyz = self._xyz[selected_pts_mask]
        # new_normal = self._normal[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # args = [new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        args = [new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        if self.use_intrinsic:
            new_reflectance = self._reflectance[selected_pts_mask]
            new_intensity = self._intensity[selected_pts_mask]
            new_residual_dc = self._residual_dc[selected_pts_mask]
            new_residual_rest = self._residual_rest[selected_pts_mask]
            new_light = self._light[selected_pts_mask]
            new_offset = self._offset[selected_pts_mask]
            args += [new_reflectance, new_intensity, new_residual_dc, new_residual_rest, new_light, new_offset]

        self.densification_postfix(*args)

    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_normal):
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):

        grads = self.xyz_gradient_accum / self.denom
        # grads_normal = self.normal_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # grads_normal[grads_normal.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent, grads_normal, max_grad_normal)
        # self.densify_and_split(grads, max_grad, extent, grads_normal, max_grad_normal)
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 透明度小于阈值的点
        if max_screen_size: # 如果max_screen_size不为空
            big_points_vs = self.max_radii2D > max_screen_size # 2D半径大于阈值的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 尺度大于阈值的点
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) # 透明度小于阈值或2D半径大于阈值或尺度大于阈值的点
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        # self.normal_gradient_accum[update_filter] += torch.norm(
        #     self.normal_activation(self._normal.grad)[update_filter], dim=-1,
        #     keepdim=True)
        self.denom[update_filter] += 1