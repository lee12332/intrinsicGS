from collections import defaultdict
import os
import torch
import torch.nn.functional as F
import torchvision
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from utils.system_utils import prepare_output_and_logger
from tqdm import tqdm
from utils.image_utils import psnr, visualize_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image, make_grid
from lpipsPyTorch import lpips
from utils.loss_utils import get_sparsity_weight

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) 
    
    # 初始化高斯和场景
    gaussians = GaussianModel(dataset.sh_degree, use_intrinsic=True if args.type == 'intrinsic' else False, geo_enhence=pipe.geo_enhence) # 主要是初始化gs的属性（为0，未赋值）
    scene = Scene(dataset, gaussians)#主要是相机初始化

    #对初始化的高斯和场景类读取数据赋值，GaussianModel初始化后gs还没有属性值和具体gs个数，create_from_XXX主要做了两件事：1.用特定值初始化（不是随机初始化）2.将属性设置为可训练对象
    if args.checkpoint: # ckpt
        print("从检查点文件创建: {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
    elif scene.loaded_iter: # ply
        ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "point_cloud.ply")
        print("从ply文件创建: {}".format(ply_path))
        gaussians.load_ply(ply_path)
    else: # colmap
        print("从colmap点云创建: {}".format(os.path.join(args.model_path, "point_cloud")))
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
    
    # 设置训练参数
    gaussians.training_setup(opt)
    
    # 设置intrinsic
    intrinsic_kwargs = dict()
    if args.type == "intrinsic":
        intrinsic_kwargs['gamma'] = opt.gamma
    
    """ Prepare render function and bg """
    render_fn = render_fn_dict[args.type] # render or intrinsic
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    # Training setup
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)#存储平滑后的指标值
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    
    # Training loop
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))#randint随机整数,viewpoint_cam是一个cam，不是所有cam

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        """ Important """
        intrinsic_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,opt=opt, is_training=True, dict_params=intrinsic_kwargs) #返回值中字典的键{"render": ,"pseudo_normal": ,"surface_xyz": ,"opacity": ,
                                                                                                                                   #"depth": ,"viewspace_points":,"visibility_filter": radii > 0,
                                                                                                                                   # "radii": ,"num_rendered": "num_contrib": ,"tb_dict":,"loss":,}
        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        loss.backward()

        with torch.no_grad():
            
            if pipe.save_training_vis: # save vis
                save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration,  intrinsic_kwargs)
            
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_intrinsic"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)
            
            # Log and save
            training_report(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=intrinsic_kwargs)
                
            # Densification
            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 20
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.densify_grad_normal_threshold)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() # 重置透明度

            # Optimizer step
            gaussians.step()
            for component in intrinsic_kwargs.values():
                try:
                    component.step()
                except:
                    pass
                
            """保存结果"""
            
            # 保存ply模型文件
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians(ply)".format(iteration))
                scene.save(iteration) 

            # 保存ckpt
            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            
            if iteration >= 5000 and iteration <= 15000 and iteration % 1000 == 0:
                lambda_offset = (((iteration - 5000)/1000 - 10)**2)/100
                if lambda_offset < 0.02:
                    lambda_offset = 0.02
                opt.lambda_offset = lambda_offset
                # opt.lambda_offset = 0.02
                print(f'迭代{iteration} offset权重：{opt.lambda_offset}')
                
            if iteration >= 5000 and iteration % 1000 == 0:
                opt.lambda_offset = 0.02
                print(f'迭代{iteration} offset权重：{opt.lambda_offset}')
            
            if iteration >= 15000 and iteration % 1000 == 0:
                opt.lambda_residual = 0.02
                print(f'迭代{iteration} residual权重：{opt.lambda_residual}')
            
def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_intrinsic_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    intrinsic_image = torch.clamp(render_pkg.get("intrinsic", torch.zeros_like(image)), 0.0, 1.0)

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    # normal = torch.clamp(
                    #     render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # intrinsic
                    reflectance = torch.clamp(render_pkg.get("reflectance", torch.zeros_like(image)), 0.0, 1.0)

                    shading = torch.clamp(render_pkg.get("shading", torch.zeros_like(image)), 0.0, 1.0)
                    if shading.shape[0] == 1:
                        shading = shading.repeat(3, 1, 1)
                    residual = render_pkg.get("residual", torch.zeros_like(image))
                    light = render_pkg.get("light", torch.zeros_like(image))

                    grid = torchvision.utils.make_grid(torch.stack([gt_image, image, 
                                                                    intrinsic_image, reflectance, 
                                                                    residual, shading, light], dim=0), nrow=3)
 
                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(intrinsic_image, gt_image).mean().double()
                    psnr_test += psnr(intrinsic_image, gt_image).mean().double()
                    psnr_intrinsic_test += psnr(intrinsic_image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_intrinsic_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_INTRINSIC {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_intrinsic_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_intrinsic', psnr_intrinsic_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_INTRINSIC {}".format(l1_test, psnr_test, psnr_intrinsic_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def save_training_vis(viewpoint_cam, gaussians, background, render_fn, 
                      pipe, opt, first_iter, iteration, intrinsic_kwargs):
    
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, dict_params=intrinsic_kwargs)
            gt_image = viewpoint_cam.original_image.cuda()
            gt_normal = viewpoint_cam.normal.cuda()
            
            visualization_list = [
                gt_image, # gt image
                # render_pkg["render"], # render image 
                gt_normal * 0.5 + 0.5,
                visualize_depth(viewpoint_cam.depth.cuda()), # gt depth
                visualize_depth(render_pkg["depth"]), # render depth
                
                # render_pkg["normal"] * 0.5 + 0.5,   # render normal
            ]

            if args.type == 'intrinsic':
                lab = viewpoint_cam.lab.cuda()
                sparsity_weight = get_sparsity_weight(gt_image, gt_normal, lab)
                
                visualization_list.extend([
                    render_pkg["intrinsic"],
                    render_pkg["reflectance"],
                    render_pkg["shading"].repeat(3, 1, 1),
                    render_pkg["residual"],
                    # intrinsic_2d,
                    sparsity_weight.repeat(3, 1, 1),
                    render_pkg["reflectance"] + render_pkg["offset"],                render_pkg["offset"],
                    ])


            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render', 'intrinsic'], default='intrinsic')
    parser.add_argument('--mode', choices=['stage', 'joint'], default='joint')
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)   

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")

