import colmap_cmd
import os
scene = 'scan124'
# 311.7691955566406,311.7691955566406,320.0,240.0
target_path = '/home/lhy/research/datasets/intrinsic/i2sdf'


scene_path = os.path.join(target_path, scene)   
created_sparse_path = os.path.join(scene_path, 'created', 'sparse')
triangulated_sparse_path = os.path.join(scene_path, 'triangulated', 'sparse')

colmap_cmd.exhaustive_matcher(scene_path)
colmap_cmd.point_triangulator(scene_path, created_sparse_path, triangulated_sparse_path)

"""
查看稀疏模型重建结果
    (1) 打开COLMAP GUI
    (2).File --> Import Model: 选择triangulated/sparse/model目录
    (3)."Directory does not contain a project.ini" 点击No: 查看重建结果
    (4).导出model为txt:File --> Export model as text: 将结果保存在triangulated/sparse/model目录下,
        会生成project.ini, cameras.txt, images.txt, points3D.txt 

"""

