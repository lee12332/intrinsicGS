#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# root_dir="/home/lhy/data1/code/intrinsic-gs/datasets/100/"
root_dir="/home/lhy/code/intrinsic/intrinsic-origin/datasets/intrinsic/ScanNet++/"
# root_dir="/home/lhy/data1/datasets/100/"
# list="chair drums ficus hotdog lego materials mic ship"
# list="0b031f3119"
list="0b031f3119 56a0ec536c 8b5caf3398 b20a261fdf f8f12e4e6b"
# list="56a0ec536c"


for i in $list; do
python train.py \
-s ${root_dir}${i} \
-m output/spp/${i} \
-t intrinsic \
--eval \
--iterations 30000 \
--compute_intrinsic_composition \
--densification_interval 100 \
--densify_until_iter 15000 \
--lambda_scale_depth 0.1 \
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


