from argparse import ArgumentParser
import os
import sys
import random
import numpy as np
import colmap_cmd
import shutil
import glob
import cv2
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


scene_seq = {'office_0': 1,
             'office_1': 2,
             'office_2': 2,
             'office_3': 1,
             'office_4': 2,
             'room_0': 2,
             'room_1': 2,
             'room_2': 1} # following intrinsic-nerf



if __name__ == "__main__":
    
    parser = ArgumentParser(description="pre-process for replica")
    parser.add_argument('--dataset_path', type=str, default='/home/lhy/research/datasets/Replica_Dataset')
    parser.add_argument('--target_path', type=str, default='/home/lhy/research/datasets/intrinsic')
    parser.add_argument("--scene", type=str, choices=['room_{}'.format(i) for i in range(3)] + ['office_{}'.format(i) for i in range(5)], default='room_2')
    parser.add_argument("--sequence", choices=[-1, 1, 2], type=int,  default=-1)
    parser.add_argument('--total_num', type=int, default=900)
    parser.add_argument('--sample_step', type=int, default=5)
    parser.add_argument("--random_sample", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    
    if not os.path.exists(args.dataset_path):
        print(f'数据集路径 {args.dataset_path} 不存在！')
        sys.exit()
        
    scene_path = os.path.join(args.dataset_path, args.scene)

    if not os.path.exists(scene_path):
        print(f'场景路径 {scene_path} 不存在！')
        sys.exit()
       
    sequence = scene_seq[args.scene] if args.sequence == -1 else args.sequence
    data_path = os.path.join(scene_path, 'Sequence_{}'.format(sequence))
    print(f"数据集路径: {data_path}")
    
    rgb_path = os.path.join(data_path, "rgb")
    depth_path = os.path.join(data_path, "depth")
    traj_file = os.path.join(data_path, "traj_w_c.txt")
    c2w_list = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
    rgb_list = sorted(glob.glob(rgb_path + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    depth_list = sorted(glob.glob(depth_path + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
    train_seq = list(range(0, args.total_num, args.sample_step))
    test_seq = [x + args.sample_step // 2 for x in train_seq] 
    target_path = os.path.join(args.target_path, 'replica', args.scene)
    print(f"目标路径: {target_path}")
    
    if os.path.exists(target_path):
        print(f"目录 '{target_path}' 已经存在，将被删除！")
        shutil.rmtree(target_path)
        print(f"目录' {target_path}' 删除成功！")

    train_image_path = os.path.join(target_path, 'images')
    train_depth_path = os.path.join(target_path, 'depths')
    
    test_image_path = os.path.join(target_path, 'test', 'images')
    test_depth_path = os.path.join(target_path, 'test', 'depths')
    
    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(train_depth_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(test_depth_path, exist_ok=True)

    print("数据集复制开始")

        
    for seq in train_seq:
        rgb_path = rgb_list[seq]
        rgb_name = rgb_path.split('/')[-1]
        depth_path = depth_list[seq]
        depth_name = depth_path.split('/')[-1]
        shutil.copy(rgb_path, train_image_path)
        shutil.copy(depth_path, train_depth_path)
        
        copy_rgb_name = os.path.join(train_image_path, rgb_name)
        copy_depth_name = os.path.join(train_depth_path, depth_name)
        
        new_rgb_name = os.path.join(train_image_path, f'rgb_{int(seq):04d}.png')
        new_depth_name = os.path.join(train_depth_path, f'depth_{int(seq):04d}.png')
        
        os.rename(copy_rgb_name, new_rgb_name)
        os.rename(copy_depth_name, new_depth_name)


    for seq in test_seq:
        rgb_path = rgb_list[seq]
        rgb_name = rgb_path.split('/')[-1]
        depth_path = depth_list[seq]
        depth_name = depth_path.split('/')[-1]
        shutil.copy(rgb_path, test_image_path)
        shutil.copy(depth_path, test_depth_path)
        
        copy_rgb_name = os.path.join(test_image_path, rgb_name)
        copy_depth_name = os.path.join(test_depth_path, depth_name)
        
        new_rgb_name = os.path.join(test_image_path, f'rgb_{int(seq):04d}.png')
        new_depth_name = os.path.join(test_depth_path, f'depth_{int(seq):04d}.png')
        # rename是因为colmap数据库的images排序是按照文件名的字典序排序的
        os.rename(copy_rgb_name, new_rgb_name)
        os.rename(copy_depth_name, new_depth_name)
        
    
    print("数据集复制完成")
    
    
    
    # print("——————————————相机内参文件开始生成——————————————————")
    
    created_sparse_path = os.path.join(target_path, 'created', 'sparse')
    os.makedirs(created_sparse_path, exist_ok=True)
    
    triangulated_sparse_path = os.path.join(target_path, 'triangulated', 'sparse')
    os.makedirs(triangulated_sparse_path, exist_ok=True)
    
    final_sparse_path = os.path.join(target_path, 'sparse', '0')
    os.makedirs(final_sparse_path, exist_ok=True)
    
    height, width, _ = imageio.imread(rgb_list[0]).shape
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = width / 2.0 / math.tan(math.radians(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    params = np.array(tuple(map(float, [fx, cx, cy])))
    
    camera = Camera(id=1, model=CAMERA_MODEL_IDS[0].model_name, width=width, height=height, params=params)
    cameras = {1: camera}
    
    # with open(os.path.join(created_sparse_path, "cameras" + '.txt'), 'w') as f:
    #     pass
    
    write_cameras_text(cameras, os.path.join(created_sparse_path, "cameras" + '.txt'))
    
    train_c2w = c2w_list[train_seq]
    train_extrinsics = []
    images = {}
    
    for image_id, c2w in enumerate(train_c2w):
        w2c = np.linalg.inv(c2w)
        # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        
        R = w2c[:3, :3]
        qvec = rotmat2qvec(R)
        T = w2c[:3, 3].reshape(3)
        tvec = T
        camera_id = 1 # rgb_{int(train_seq[image_id]):04d}.png
        image_name = f'rgb_{int(train_seq[image_id]):04d}.png'
        # print(image_name)
        images[image_id] = Image(
            id=image_id+1,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=None,
            point3D_ids=None,
        )
        
        
    
    test_c2w = c2w_list[test_seq]
    test_extrinsic = []
    for c2w in test_c2w:
        w2c = np.linalg.inv(c2w)
        # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        R = w2c[:3, :3]
        T = w2c[:3, 3].reshape(3)
        qvec = rotmat2qvec(R)
        extrinsics = np.concatenate([qvec, T])
        test_extrinsic.append(extrinsics)
        
    write_images_text(images, os.path.join(created_sparse_path, "images" + '.txt'))

    print("——————————————相机外参文件生成完成——————————————————")
    
    with open(os.path.join(created_sparse_path, "points3D" + '.txt'), 'w') as f:
        pass
    
    

    
    print("——————————————开始执行SfM——————————————————")
    # colmap_cmd.SfM(target_path)
    
    colmap_cmd.feature_extractor(target_path)
    # 1 SIMPLE_PINHOLE 640 480 320.00000000000006,319.5,239.5
    # 手动修改或者连接数据库
    # 然后执行next.py
    # colmap_cmd.exhaustive_matcher(target_path)
    # colmap_cmd.point_triangulator(target_path, created_sparse_path, triangulated_sparse_path)
    
    
    
    print("——————————————SfM执行结束——————————————————")
