import os
import sys
import subprocess

def automatic_reconstructor(dataset_path):
    assert os.path.exists(os.path.join(dataset_path, "images")), "images folder not found in the dataset path."
    cmd = ["colmap", "automatic_reconstructor", 
            "--workspace_path", dataset_path, 
            "--image_path", os.path.join(dataset_path, "images"),
            "--no_gui"]
    try:
        subprocess.run(cmd, check=True)
        print("自动重建执行完成")
    except subprocess.CalledProcessError as e:
        print("自动重建执行失败:", e)
    

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

def exhaustive_matcher(dataset_path):
    cmd = ["colmap", "exhaustive_matcher", 
           "--database_path", os.path.join(dataset_path, "database.db"),
        #    "--SiftMatching.guided_matching", "1",
           ]
    try:
        subprocess.run(cmd, check=True)
        print("特征匹配执行完成")
    except subprocess.CalledProcessError as e:
        print("特征匹配执行失败:", e)
        
"""
colmap point_triangulator \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images
    --input_path path/to/manually/created/sparse/model \
    --output_path path/to/triangulated/sparse/model
"""


def point_triangulator(dataset_path, created_sparse_path, triangulated_sparse_path):
    cmd = ['colmap', 'point_triangulator',
           '--database_path', os.path.join(dataset_path, 'database.db'),
           '--image_path', os.path.join(dataset_path, 'images'),
           '--input_path', created_sparse_path,
           '--output_path', triangulated_sparse_path]
    try:
        subprocess.run(cmd, check=True)
        print("点三角化执行完成")
    except subprocess.CalledProcessError as e:
        print("点三角化执行失败:", e)
    

def mapper(dataset_path):
    cmd = ["colmap", "mapper", 
           "--database_path", os.path.join(dataset_path, "database.db"), 
           "--image_path", os.path.join(dataset_path, "images"), 
           "--output_path", os.path.join(dataset_path, "sparse")]
    try:
        subprocess.run(cmd, check=True)
        print("稀疏建图执行完成")
    except subprocess.CalledProcessError as e:
        print("稀疏建图执行失败:", e)

def SfM(dataset_path):
    print("开始执行SfM")
    feature_extractor(dataset_path)
    exhaustive_matcher(dataset_path)
    os.makedirs(os.path.join(dataset_path, 'sparse'), exist_ok=True)
    mapper(dataset_path)
    print("SfM执行完成")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help="the path contains the folder of images")
    parser.add_argument('--auto', action="store_true", help="whether use colmap automatic reconstructor")
    parser.add_argument('--SfM', action="store_true", help="whether use colmap SfM")
    args = parser.parse_args(sys.argv[1:])
    
    SfM(dataset_path=args.dataset_path)
    

