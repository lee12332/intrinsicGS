import time
from scene.cameras import Camera
import torch
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import cv2
from utils.image_utils import rgb2lab, rgb2lab_np


class Cluster:
    def __init__(self, lab_weights =  [0.3,   1,   1],
                       xyz_weights =  [0.3, 0.3, 0.3],
                       normal_weights=[0.2, 0.2, 0.2], 
                       use_xyz=True, 
                       use_normal=False, 
                       predict_device='cuda'):
        self.lab_weights = lab_weights
        self.xyz_weights = xyz_weights
        self.normal_weights = normal_weights
        self.scaler = None
        self.use_xyz = use_xyz
        self.use_normal = use_normal
        self.cluster_method = None
        self.cluster_mean = None # {'label': mean reflectance value}
        self.target_reflectance = {} # {cam_id: target_reflectance}
        self.target_scale = {}
        self.predict_device = predict_device
        if use_xyz:
            self.weights = lab_weights + xyz_weights
            if use_normal:
                self.weights += normal_weights
        elif use_normal:
            self.weights = lab_weights + normal_weights
        else:
            self.weights = lab_weights # 不想优化了
    

    # 反射率聚类
    def reflectance_cluster(self, reflectance_list, camera_list=None, depth_list=None, 
                                  down_step=2, cluster_method='meanshift', **kwargs):
        # 范围 0-1
        num_images = len(reflectance_list)
        width = reflectance_list[0].shape[2] // down_step
        height = reflectance_list[0].shape[1] // down_step
        
        features = []
        reflectances = []
        
        if depth_list is None:
            depth_list = [None] * num_images
        
        
        for reflect, cam, depth in zip(reflectance_list, camera_list, depth_list):
            reflectance_feature = self.cal_cluster_feature(reflect, cam, depth, down_step)
            features.append(reflectance_feature)
            # 特征没有标准化，只是提取出了lab和position
            
            reflectances.append(reflect.permute(1, 2, 0)[::down_step, ::down_step].cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        print('特征维度：', features.shape[1])
        features = self.normalize_feature(features) # 标准化特征并将mean和std给了类
        reflectances = np.stack(reflectances, axis=0)
        
        if cluster_method == 'kmeans':
            print('使用kmeans聚类')
            n_clusters = kwargs.get('n_clusters', 30)
            self.cluster_method = KMeans(n_clusters)
        else:
            print('使用meanshift聚类')
            quantile = kwargs.get('quantile', 0.3)
            n_samples = kwargs.get('n_samples', 5000)
            band_factor = kwargs.get('band_factor', 0.4)
            bw = estimate_bandwidth(features, quantile=quantile, n_samples=n_samples)
            bw = max(bw * band_factor,0.01)
            print(f"带宽估计: {bw}")
            self.cluster_method = MeanShift(bandwidth=bw, bin_seeding=True)
        # self.ms = KMeans(10)
        print('------开始聚类------')
        start_time = time.time()
        
        labels = self.cluster_method.fit_predict(features)
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        end_time = time.time()
        print('------结束聚类------')
        print(f"聚类数量 : {n_clusters}")
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        print(f"聚类时间：{minutes} 分钟 {seconds:.2f} 秒")      
        labels = labels.reshape(num_images, height, width)
        
        
        # 构建提速网格，加速预测速度 as intrinsicNeRF did
        # OOM!
        # da ka ra 我们的想法是
        # 1. 先完成聚类
        # 2. 然后在训练的时候，对于特定相机渲染出来的反射率
        # 3. 如果cluster里的target_image[cam_id] 为空
        #      (1)在gpu上计算反射率特征，得到 [h, w, 6] 的特征矩阵
        #      (2)计算特征到centers的距离，得到label
        #      (3)得到labels后，得到target_img，然后保存在target_image[cam_id]
        # 4. 否则直接与target_image[cam_id]对比
        
        feature_centers = self.cluster_method.cluster_centers_

        reflectance_centers = np.zeros((n_clusters, 3), dtype=np.float32)
        
        for i, center in enumerate(feature_centers):
            mask = np.where(labels == i, True, False)
            reflectance_centers[i] = np.mean(reflectances[mask], axis=0)

        self.feature_centers = torch.from_numpy(feature_centers).to(self.predict_device)
        self.reflectance_centers = torch.from_numpy(reflectance_centers).to(self.predict_device)
        # print('反射率中心点：\n', self.reflectance_centers)
        self.scale_reflectance_centers = self.reflectance_centers.mean(dim=-1, keepdim=True)
        

    def cal_cluster_feature(self, reflectance, camera, depth, down_step=1, mode='fit'):
        
        feature_list = []
        reflectance = reflectance.permute(1, 2, 0)[::down_step, ::down_step].reshape(-1, 3) # [N, 3]
        lab = rgb2lab(reflectance)
        feature_list.append(lab)
        
        if self.use_xyz:
            if depth is None:
                position = camera.unproject(gt_depth=True)
            else:
                position = camera.unproject(splatted_depth=depth)
            
            position = position[::down_step, ::down_step].reshape(-1, 3)
            
            feature_list.append(position)
            
  
        if self.use_normal:
            normal = camera.normal.permute(1, 2, 0)[::down_step, ::down_step].reshape(-1, 3)
            feature_list.append(normal)
        
        if mode == 'fit':
            cluster_feature = [f.cpu().numpy() for f in feature_list]
            cluster_feature = np.concatenate(cluster_feature, axis=-1)

        else: # predict
            cluster_feature = [f.to(self.predict_device) for f in feature_list]
            cluster_feature = torch.cat(feature_list, dim=-1)
        return cluster_feature
    
    
    # 用于fit
    def normalize_feature(self, feature, mode='fit'):
        # 对特征进行归一化
        # 然后加权
        """
        scaler.fit(data): 仅计算数据的均值和标准差，不进行转换操作。
        scaler.transform(data): 根据已经训练好的模型或转换器，对输入的新数据 new_data 进行相应的转换操作，测试
        scaler.fit_transform(data): 计算数据的均值和标准差，并对数据进行标准化处理,用于训练模型并对数据进行转换
        scaler.inverse_transform(normalized_data): 将标准化后的数据转换回原始数据空间（反向操作）。
        """
        
        assert feature.shape[1] == len(self.weights), f'特征维度不匹配，特征维度为{feature.shape[1]}, 权重维度为{len(self.weights)}'
        
        self.scaler = StandardScaler()
        
        if mode == 'fit':
            feature = self.scaler.fit_transform(feature)
            self.feature_mean = torch.from_numpy(self.scaler.mean_).to(self.predict_device)
            self.feature_std = torch.from_numpy(self.scaler.scale_).to(self.predict_device)
            weighted_feature = [feature[:, i] * self.weights[i] for i in range(feature.shape[1])]
            weighted_feature = np.stack(weighted_feature, axis=-1)
        else:
            feature = (feature - self.feature_mean[None]) / self.feature_std[None]
            weighted_feature = [feature[:, i] * self.weights[i] for i in range(feature.shape[1])]
            weighted_feature = torch.stack(weighted_feature, dim=-1)
        
        return weighted_feature
    
    

    # 用于predict
    def predict_target_reflectance(self, reflectance, camera, depth=None):
        
        if self.target_reflectance.get(camera.uid) is not None:
            return self.target_reflectance[camera.uid]
        
        with torch.no_grad():
            print(f'相机 {camera.uid} 未找到目标反射率图像，开始预测')
            _, height, width = reflectance.shape
            feature = self.cal_cluster_feature(reflectance, camera, depth, down_step=1, mode='predict')
            feature = self.normalize_feature(feature, mode='predict')
            
            label = self.cal_label(feature)
            label = label.reshape(height, width)
            target_img = self.label2target(label).permute(2, 0, 1)
            self.target_reflectance[camera.uid] = target_img
            
            return target_img # 3 h w
    
    
    def cal_label(self, feature):
        num_f, dim = feature.shape
        feature = feature.reshape(num_f, 1, dim)
        centers = self.feature_centers[None]
        distance2 = torch.norm(feature - centers, dim=-1)
        labels = torch.argmin(distance2, dim=-1)
        return labels
    
    
    def label2target(self, label):
        height, width = label.shape
        target_img = torch.zeros((height, width, 3), dtype=torch.float32).to(label.device)
        for i, center in enumerate(self.reflectance_centers):
            mask = torch.where(label == i, True, False)
            target_img[mask] = center
        
        return target_img
    
    
    # def predict_target_scale(self, reflectance, camera, depth=None):
        
    #     if self.target_scale.get(camera.uid) is not None:
    #         return self.target_scale[camera.uid]
        
    #     # print(f'相机 {camera.uid} 未找到目标反射率强度，开始预测')
    #     _, height, width = reflectance.shape
    #     feature = self.cal_cluster_feature(reflectance, camera, depth, down_step=1, mode='predict')
    #     feature = self.normalize_feature(feature, mode='predict')
        
    #     label = self.cal_label(feature)
    #     label = label.reshape(height, width)
    #     target_img = self.label2scale(label)
    #     self.target_scale[camera.uid] = target_img
        
    #     return target_img
    

    # def label2scale(self, label):
    #     height, width = label.shape
    #     target_scale = torch.zeros((height, width), dtype=torch.float32)
    #     for i, center in enumerate(self.scale_reflectance_centers):
    #         mask = torch.where(label == i, True, False)
    #         target_scale[mask] = center
    #     return target_scale

    
    def reset_cluster(self):
        self.cluster_method = None
        self.cluster_mean = None
        self.feature_centers = torch.empty(0)
        self.reflectance_centers = torch.empty(0)
        self.feature_mean = torch.empty(0)
        self.feature_std = torch.empty(0)
        self.target_reflectance = {}
        self.target_scale = {}
        
    # def predict_reflectance_intensity(self, camera):
    #     labels = self.predict_reflectance_label(camera)
        
    
    # def visualize_reflectance_cluster(self, camera, reflectance, depth):
    #     label = self.predict_reflectance_label(camera, reflectance, depth)
        
    #     label_unique = np.unique(label)
    #     visual_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.float32)
    #     for k in label_unique:
    #         mask = np.where(label == k, True, False)
    #         visual_image[mask] = self.cluster_mean[k]
    #     return visual_image
    
