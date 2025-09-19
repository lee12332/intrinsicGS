#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0

# root_dir="/home/lhy/data1/code/intrinsic-gs/datasets/100/"
root_dir="datasets/intrinsic/replica/"
# root_dir="/home/lhy/data1/datasets/100/"
# list="chair drums ficus hotdog lego materials mic ship"
# list=" office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 room_1 room_2 office_2 office_3 office_4"
    #   16:02    35:21    34:05  606062    539912   749867  714278 467988
    #  367283   707879    582203  34:48    33:17   39:11   38:45  25:17

# list="office_3 office_4"
list="room_0"


for i in $list; do
python train.py \
-s ${root_dir}${i}/train \
-m output/replica/${i} \
-t intrinsic \
--iterations 30000 \
--compute_intrinsic_composition \
--densification_interval 100 \
--densify_until_iter 15000 \
--lambda_depth 0 \
--lambda_mask_entropy 0 \
--lambda_vanilla 0 \
--lambda_intrinsic 1 \
--lambda_reflectance_sparsity 0.2 \
--lambda_shading_smooth 1 \
--use_residual \
--lambda_residual 1 \
--lambda_prior_chromaticity 1 \
--lambda_offset 1 \
--save_training_vis \
--save_training_vis_iteration 1000

done



# conda create -n intrinsic python=3.8
# conda activate intrinsic
# . ~/tools/switch-cuda.sh 11.8
# pip install matplotlib==3.5.2 numpy==1.21.5 tensorboard==2.10.0 tqdm==4.64.1 imageio==2.31.2 opencv-python==4.1.0.25 pillow==9.5.0 plyfile==0.9 scipy==1.7.3
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install ./submodules/simple-knn/
# pip install kornia==0.6.12
# pip install git+https://github.com/jamesbowman/openexrpython.git
# pip install git+https://github.com/tvogels/pyexr.git
