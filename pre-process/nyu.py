from argparse import ArgumentParser
import os
import sys
import random
import numpy as np
import colmap_cmd
import shutil

if __name__ == "__main__":
    
    parser = ArgumentParser(description="pre-process for replica")
    parser.add_argument('--source_path', type=str, default='/home/lhy/research/datasets/Replica_Dataset')
    parser.add_argument('--target_path', type=str, default='/home/lhy/research/datasets/intrinsic')
    parser.add_argument("--scene", type=str, choices=['room_{}'.format(i) for i in range(3)] + ['office_{}'.format(i) for i in range(5)], default='office_3')
    parser.add_argument("--sequence", choices=[1, 2], type=int,  default=2)
    parser.add_argument('--total_num', type=int, default=900)
    parser.add_argument('--sample_step', type=int, default=5)
    parser.add_argument("--random_sample", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    
    print(f"源文件路径: {args.source_path}")
    print(f"目标文件路径: {args.target_path}")
    
    scene_path = os.path.join(args.source_path, args.scene)


    
    if not os.path.exists(scene_path):
        print('{}不存在！'.format(args.scene))
        sys.exit()
    
    
    train_seq = list(range(0, args.total_num, args.sample_step))
    
    # if args.random_sample:
    #     train_seq = random.sample(range(args.image_all_num), 50)
    # else:
    #     train_seq = list(np.floor(np.linspace(0, args.image_all_num-1, args.image_sample_num)).astype(int))
    
    scene_path = os.path.join(scene_path, 'Sequence_{}'.format(args.sequence))
    
    rgb_names = [os.path.join(scene_path, 'rgb', 'rgb_{}.png'.format(i)) for i in train_seq]
    depth_names = [os.path.join(scene_path, 'depth', 'depth_{}.png'.format(i)) for i in train_seq]
    # test_names = [os.path.join(scene_path, 'rgb', 'rgb_{}.png'.format(i)) for i in test_seq]
    
    target_path = os.path.join(args.target_path, 'replica', args.scene)
    
    try:
        shutil.rmtree(target_path)
        print(f"Folder '{target_path}' and its contents deleted successfully")
    except OSError as e:
        print(f"Error: {target_path} : {e.strerror}")
    
    
    image_path = os.path.join(target_path, 'images')
    depth_path = os.path.join(target_path, 'depths')
    # test_path = os.path.join(target_path, 'tests')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    # os.makedirs(test_path, exist_ok=True)
    
    for rgb, depth in zip(rgb_names, depth_names):
        shutil.copy(rgb, image_path)
        shutil.copy(depth, depth_path)
        # shutil.copy(test, test_path)
        
    print("数据集采样完成")
    # print("——————————————开始执行SfM——————————————————")
    # # colmap_cmd.SfM(target_path)
    # print("——————————————SfM执行结束——————————————————")
