# intrinsicGS
Official repository of “Multi-view intrinsic decomposition of indoor scenes under a 3D Gaussian splatting framework”

### <p align="center">[🌐Project Page](https://github.com/lee12332/intrinsicGS.git) | [📰Paper](https://www.cjig.cn/zh/article/doi/10.11834/jig.240505/)</p>

### Installation
#### Clone this repo
```shell
git clone https://github.com/lee12332/intrinsicGS.git
```
#### Install dependencies
```shell
# install environment
conda env create --file intrinsic_env.yml
conda activate intrinsic

# install pytorch=1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# install torch_scatter==2.1.1
pip install torch_scatter==2.1.1

# install kornia==0.6.12
pip install kornia==0.6.12

# install nvdiffrast=0.3.1
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

#### Install the pytorch extensions
We recommend that users compile the extension with CUDA 11.8 to avoid the potential problems mentioned in [3D Guassian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

```shell
# install knn-cuda
pip install ./submodules/simple-knn

# install bvh
pip install ./bvh

# install relightable 3D Gaussian
pip install ./r3dg-rasterization
```


### Data preparation
#### Data Structure
We organize the datasets like this:
```
intrinsicGS
── datasets
    ── intrinsic
       ── replica
            ── office_0
                ── test
                ── train
                    ── images
                        ── rgb_0000.png
                        ...
                        ── rgb_1999.png
                    ── depths
                        ── depth_0000.png
                        ...
                        ── depth_1999.png
                    ── sparse
                        ── 0
                            ── cameras
                            ── images
                            ── points3D
    
```

### Running
We run the code in a single NVIDIA GeForce RTX 3090 GPU (24G). To reproduce the results in the paper, please run the following code.
replica:
```
python train.py -s datasets/intrinsic/replica/office_0/train
                -m output/replica/office_0
                -t intrinsic
                --iterations 30000
                --compute_intrinsic_composition
                --densification_interval 100
                --densify_until_iter 15000
                --lambda_depth 0
                --lambda_mask_entropy 0
                --lambda_vanilla 0
                --lambda_intrinsic 1
                --lambda_reflectance_sparsity 0.2
                --lambda_shading_smooth 1
                --use_residual
                --lambda_residual 1
                --lambda_prior_chromaticity 1
                --lambda_offset 1
                --save_training_vis
                --save_training_vis_iteration 1000
```

### Citation
If you find our work useful in your research, please be so kind to cite:
```
@article{IGS2025,
    author    = {Lyu Hengye， Liu Yanli， Li Hong， Yuan Xia， Xing Guanyu},
    title     = {Multi-view intrinsic decomposition of indoor scenes under a 3D Gaussian splatting framework},
    journal   = {Journal of Image and Graphics， 30(7):2514-2527 DOI： 10.11834/jig.240505},
    year      = {2025},
}
```
