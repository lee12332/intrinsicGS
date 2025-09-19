from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import random
import cv2
import numpy as np
import tqdm
import colmap_cmd
import shutil
import glob
# import cv2
import imageio.v2 as imageio
import math
import collections
import subprocess

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    # if len(images) == 0:
    #     mean_observations = 0
    # else:
    #     mean_observations = sum(
    #         (len(img.point3D_ids) for _, img in images.items())
    #     ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        # + "# Number of images: {}, mean observations per image: {}\n".format(
        #     len(images), mean_observations
        # )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            # points_strings = []
            # for xy, point3D_id in zip(img.xys, img.point3D_ids):
            #     points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            # fid.write(" ".join(points_strings) + "\n")
            fid.write("\n")
            
def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_model(cameras, images, points3D, path, ext=".txt"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        pass
        # write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        # write_images_binary(images, os.path.join(path, "images" + ext))
        # write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def feature_extractor(dataset_path, camera_model='SIMPLE_PINHOLE', specify_intrinsic=None):
    cmd = ["colmap", "feature_extractor", 
           "--database_path", os.path.join(dataset_path, "database.db"), 
           "--image_path", os.path.join(dataset_path, "images"), 
           "--ImageReader.single_camera", "1",
           "--ImageReader.camera_model", camera_model]
    if specify_intrinsic:
        pass
    try:
        subprocess.run(cmd, check=True)
        print("特征提取执行完成")
    except subprocess.CalledProcessError as e:
        print("特征提取执行失败:", e)


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []


    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fl_x = contents['fl_x']
        fl_y = contents['fl_y']
        

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            image_name = Path(image_path).stem
            
            depth_path = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # image, is_hdr = load_img(image_path)

            # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # image_mask = np.ones_like(image[..., 0])
            # if image.shape[-1] == 4:
            #     image_mask = image[:, :, 3]
            #     image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])

            # read depth and mask
            # depth = None
            # normal = None


            # fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=image_mask,
            #                             image_path=image_path, depth=depth, normal=normal, image_name=image_name,
            #                             width=image.shape[1], height=image.shape[0], hdr=is_hdr,))

            # if debug and idx >= 5:
            #     break

    return cam_infos


  


if __name__ == "__main__":
    
    parser = ArgumentParser(description="pre-process for replica")
    parser.add_argument('--dataset_path', type=str, default='/home/lhy/research/datasets/i2sdf/synthetic')
    parser.add_argument('--target_path', type=str, default='/home/lhy/research/datasets/intrinsic')
    parser.add_argument("--scene", type=str, default='scan124')
    parser.add_argument('--total_num', type=int, default=267)
    parser.add_argument('--sample_step', type=int, default=1)
    parser.add_argument("--random_sample", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    
    # if not os.path.exists(args.dataset_path):
    #     print(f'数据集路径 {args.dataset_path} 不存在！')
    #     sys.exit()
        
    scene_path = os.path.join(args.dataset_path, args.scene)

    # if not os.path.exists(scene_path):
    #     print(f'场景路径 {scene_path} 不存在！')
    #     sys.exit()
    
    data_path = scene_path
    # print(f"数据集路径: {data_path}")
    # rgb_path = os.path.join(data_path, "image")
    depth_path = os.path.join(data_path, "depth")
    # trans_file = os.path.join(data_path, "transforms_train.json")
    # # c2w_list = np.loadtxt(trans_file, delimiter=" ").reshape(-1, 4, 4)
    # rgb_list = sorted(glob.glob(rgb_path + '/*.png'), key=lambda file_name: int(file_name.split('/')[-1].split(".")[0]))
    depth_list = sorted(glob.glob(depth_path + '/*.exr'), key=lambda file_name: int(file_name.split('/')[-1].split(".")[0]))
    train_seq = list(range(0, args.total_num, args.sample_step))
    target_path = os.path.join(args.target_path, 'i2sdf', args.scene)
    # print(f"目标路径: {target_path}")
    
    # if os.path.exists(target_path):
    #     print(f"目录 '{target_path}' 已经存在，将被删除！")
    #     shutil.rmtree(target_path)
    #     print(f"目录' {target_path}' 删除成功！")

    # train_image_path = os.path.join(target_path, 'images')
    train_depth_path = os.path.join(target_path, 'depths')
    
    # os.makedirs(train_image_path, exist_ok=True)
    # os.makedirs(train_depth_path, exist_ok=True)

    # print("数据集复制开始")

        
    for seq in train_seq:
        # rgb_path = rgb_list[seq]
        # rgb_name = rgb_path.split('/')[-1]
        depth_path = depth_list[seq]
        depth_name = depth_path.split('/')[-1]
        # shutil.copy(rgb_path, train_image_path)
        shutil.copy(depth_path, train_depth_path)
        
    #     copy_rgb_name = os.path.join(train_image_path, rgb_name)
        copy_depth_name = os.path.join(train_depth_path, depth_name)
        
    #     new_rgb_name = os.path.join(train_image_path, f'rgb_{int(seq):04d}.png')
        new_depth_name = os.path.join(train_depth_path, f'depth_{int(seq):04d}.exr')
        
    #     os.rename(copy_rgb_name, new_rgb_name)
        os.rename(copy_depth_name, new_depth_name)


    
    # print("数据集复制完成")
    
    # data_path = '/home/lhy/research/datasets/intrinsic/i2sdf/scan124/'
    # image_path = os.path.join(data_path, 'images')
    # depth_path = os.path.join(data_path, 'depths')
    # trans_path = os.path.join(data_path, 'transforms_train.json')
    # print(trans_path)
    
    # rgb_list = sorted(glob.glob(image_path + '/rgb*.png'), key=lambda file_name: int(file_name.split('/')[-1].split(".")[0].split("_")[1]))
    # depth_list = sorted(glob.glob(depth_path + '/depth*.exr'), key=lambda file_name: int(file_name.split('/')[-1].split(".")[0].split("_")[1]))
    
    # print("——————————————相机内参文件开始生成——————————————————")
    
    # created_sparse_path = os.path.join(data_path, 'created', 'sparse')
    # os.makedirs(created_sparse_path, exist_ok=True)
    
    # triangulated_sparse_path = os.path.join(data_path, 'triangulated', 'sparse')
    # os.makedirs(triangulated_sparse_path, exist_ok=True)
    
    # final_sparse_path = os.path.join(data_path, 'sparse', '0')
    # os.makedirs(final_sparse_path, exist_ok=True)
    
    # images = {}
    
    # with open(trans_path) as json_file:
    #     contents = json.load(json_file)
    #     fx = contents['fl_x']
    #     fy = contents['fl_y']
    #     width = contents['w']
    #     height = contents['h']
    #     cx = width / 2.0
    #     cy = height / 2.0
    #     params = np.array(tuple(map(float, [fx, fy, cx, cy])))
        
    #     camera = Camera(id=1, model=CAMERA_MODEL_IDS[1].model_name, width=width, height=height, params=params)
    #     cameras = {1: camera}
    


    #     frames = contents["frames"]
        
    #     for idx, frame in enumerate(frames):
    #         # NeRF 'transform_matrix' is a camera-to-world transform
    #         c2w = np.array(frame["transform_matrix"])
    #         # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    #         c2w[:3, 1:3] *= -1

    #         # get the world-to-camera transform and set R, T
    #         w2c = np.linalg.inv(c2w)
    #         # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #         R = w2c[:3, :3]
    #         qvec = rotmat2qvec(R)
    #         T = w2c[:3, 3].reshape(3)
    #         tvec = T
    #         camera_id = 1
    #         image_name = f'rgb_{int(idx):04d}.png'
            
    #         images[idx] = Image(
    #             id=idx+1,
    #             qvec=qvec,
    #             tvec=tvec,
    #             camera_id=camera_id,
    #             name=image_name,
    #             xys=None,
    #             point3D_ids=None,)


    # write_cameras_text(cameras, os.path.join(created_sparse_path, "cameras" + '.txt'))
    # write_images_text(images, os.path.join(created_sparse_path, "images" + '.txt'))

    # with open(os.path.join(created_sparse_path, "points3D" + '.txt'), 'w') as f:
    #     pass
    



    
    # print("——————————————开始执行SfM——————————————————")
    # colmap_cmd.SfM(target_path)
    
    # colmap_cmd.feature_extractor(data_path)
    # # 1 SIMPLE_PINHOLE 640 480 320.00000000000006,319.5,239.5
    # # 手动修改或者连接数据库
    # # 然后执行next.py
    # # colmap_cmd.exhaustive_matcher(target_path)
    # # colmap_cmd.point_triangulator(target_path, created_sparse_path, triangulated_sparse_path)
    
    
    
    # print("——————————————SfM执行结束——————————————————")
