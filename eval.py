from argparse import ArgumentParser
from collections import defaultdict
import os
from random import randint
import sys
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from scene import EvalScene
from gaussian_renderer import render_fn_dict
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
import torch
import numpy as np
from utils.image_utils import psnr, visualize_depth
from utils.loss_utils import ssim

"""
图片展示 
room_0 rgb_0102  rgb_0402  rgb_0457 rgb_0612 rgb_0687  rgb_0772  rgb_0772 rgb_0787
room_1 rgb_0237, rgb_0297, rgb_0382,rgb_0517 rgb_0767, rgb_0417, rgb_0732, rgb_0637
"""

def eval(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams):
    gaussians = GaussianModel(dataset.sh_degree, use_intrinsic=True if args.type == 'intrinsic' else False, geo_enhence=pipe.geo_enhence) # 球谐阶数为3
    scene = EvalScene(dataset, gaussians, test_path=args.test_path)
    print("从检查点文件创建: {}".format(args.checkpoint))
    first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
    
    intrinsic_kwargs = dict()
    if args.type == "intrinsic":
        intrinsic_kwargs['gamma'] = opt.gamma
    
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    eval_cameras = scene.getTrainCameras().copy()
    print(f"eval_cameras: {len(eval_cameras)}")
    
    
    progress_bar = tqdm(range(1, len(eval_cameras) + 1), desc="Eval progress",
                        initial=1, total=len(eval_cameras))
    
    psnr_eval = 0.0
    ssim_eval = 0.0
    lpips_eval = 0.0
    
    os.makedirs(os.path.join(args.save_path, 'intrinsic'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'reflectance'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'residual'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'offset'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'shading'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'abs_offset'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'reflectance_offset'), exist_ok=True)
    
    
    for iteration in progress_bar:
        # print(iteration)
        # print(f"iteration: {iteration}")
        viewpoint_stack = eval_cameras.copy()
            
        viewpoint_cam = viewpoint_stack[iteration -1 ]
        intrinsic_kwargs["iteration"] = iteration - first_iter
        
        results = render_fn(viewpoint_cam, gaussians, pipe, background,
                                opt=opt, is_training=False, dict_params=intrinsic_kwargs) # TODO
        


        image = results["intrinsic"]
        image = torch.clamp(image, 0.0, 1.0)
        gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)
        depth = torch.clamp(visualize_depth(results["depth"]), 0.0, 1.0)
        
        psnr_eval += psnr(image, gt_image).mean().double()
        ssim_eval += ssim(image, gt_image).mean().double()
        lpips_eval += lpips(image, gt_image, net_type='vgg').mean().double()
        
        # if viewpoint_cam.image_name in show_list:
            
        
        if (iteration-1) % 1 == 0 :
            # print(iteration)
            save_image(image, os.path.join(args.save_path, "intrinsic", f"{viewpoint_cam.image_name}.png"))
            save_image(gt_image, os.path.join(args.save_path, "gt", f"{viewpoint_cam.image_name}.png"))
            save_image(depth, os.path.join(args.save_path, "depth", f"{viewpoint_cam.image_name}.png"))
            save_image(results["reflectance"], os.path.join(args.save_path, "reflectance", f"{viewpoint_cam.image_name}.png"))
            save_image(results["shading"], os.path.join(args.save_path, "shading", f"{viewpoint_cam.image_name}.png"))
            save_image(results["offset"], os.path.join(args.save_path, "offset", f"{viewpoint_cam.image_name}.png"))
            save_image(results["offset"].abs(), os.path.join(args.save_path, "abs_offset", f"{viewpoint_cam.image_name}.png"))
            save_image(results["residual"], os.path.join(args.save_path, "residual", f"{viewpoint_cam.image_name}.png"))
            save_image(torch.clamp(results["reflectance"]+results["offset"], 0, 1), os.path.join(args.save_path, "reflectance_offset", f"{viewpoint_cam.image_name}.png"))
        
    
    psnr_eval /= len(eval_cameras)
    ssim_eval /= len(eval_cameras)
    lpips_eval /= len(eval_cameras)
    with open(os.path.join(args.save_path, "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_eval}\n")
        f.write(f"ssim: {ssim_eval}\n")
        f.write(f"lpips: {lpips_eval}\n")
    print("\nPSNR {} SSIM {} LPIPS {}".format(psnr_eval, ssim_eval, lpips_eval))

    

    

    
if __name__ == "__main__":

    print(torch.cuda.current_device())
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--test_path', type=str, default='/home/lhy/code/intrinsic/intrinsic-origin/datasets/intrinsic/replica/room_0/test')
    parser.add_argument('--save_path', type=str, default='/home/lhy/code/intrinsic/intrinsic-origin/output/replica/room_0')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('-t', '--type', choices=['intrinsic'], default='intrinsic')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default='/home/lhy/code/intrinsic/intrinsic-origin/output/replica/room_0/chkpnt30000.pth')
    args = parser.parse_args(sys.argv[1:])
    print(f"eval model path: {args.test_path}")
    print(f"Current rendering type:  {args.type}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    with torch.no_grad():
        eval(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nEval complete.")
