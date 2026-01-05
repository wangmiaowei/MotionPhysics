import os
import point_cloud_utils as pcu
import sys
sys.path.append("gaussian-splatting")
import gc
import argparse
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp
# 启用详细cd日志输出和警告信息
wp.config.verbose = True
wp.config.verbose_warnings = True
# Particle filling dependencies
from particle_filling.filling import *
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *

from utils.save_video import save_video
from utils.threestudio_utils import cleanup




wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)
class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1):
    # Find checkpoint
    """
    if "PhysDreamer_benchmark" in model_path and "pgsr_output_res" not in model_path:
        checkpt_path = os.path.join(model_path, "point_cloud.ply")
    else:
    """

    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)

    gaussians.load_ply(checkpt_path)
    return gaussians
def evaluate_mp4(mpm_solver,mpm_init_pos,mpm_init_vol,mpm_init_cov,
                 batch,stage_num,frame_per_stage,camera_params,gaussians,
                 pipeline,background,height,width,args,force_info,render_force):

    # processing these frames
    substep_perframe = time_params["substep_perframe"]
    frame_num = time_params["frame_num"]
    total_time = time_params["total_time"]
    substep_dt = total_time / (frame_num*substep_perframe)
    hah_image_prompt = None
    if batch % stage_num == 0:

        mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol,mpm_init_cov) 
        with torch.no_grad():
            if mpm_solver.mpm_model.material not in [5,7]:
                mpm_solver.finalize_mu_lam()

            #for frame in tqdm(range(frame_num+1)):
            for frame in tqdm(range(int(stage_num*frame_per_stage*1.2))):
                delta_r = camera_params["delta_r"]
                if 'alocasia' in args.model_path:
                    delta_r = camera_params["delta_r"] if frame < (frame_num+1)/2 else camera_params["delta_r"] / frame * ((stage_num * frame_per_stage)-frame)
                        
                current_camera,camera_view_info = get_camera_view(
                    model_path,
                    default_camera_index=camera_params["default_camera_index"],
                    center_view_world_space=viewpoint_center_worldspace,
                    observant_coordinates=observant_coordinates,
                    show_hint=camera_params["show_hint"],
                    init_azimuthm=camera_params["init_azimuthm"],
                    init_elevation=camera_params["init_elevation"],
                    init_radius=camera_params["init_radius"],
                    move_camera=camera_params["move_camera"],
                    current_frame=frame,
                    delta_a=camera_params["delta_a"],
                    delta_e=camera_params["delta_e"],
                    delta_r=delta_r,
                    downsample=args.downsample
                )
                rasterize = initialize_resterize(
                    current_camera, gaussians, pipeline, background
                )

                if frame!=0:
                    for substep_local in range(substep_perframe):
                        mpm_solver.p2g2p(substep_dt, device=device)


                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                rot = mpm_solver.export_particle_R_to_torch()
                cov3D = mpm_solver.export_particle_cov_to_torch()
                cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:gs_num].to(device)
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos,shif_no), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                
                cov3D = cov3D / (scale_origin * scale_origin)
                cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                opacity = opacity_render
                shs = shs_render

                if preprocessing_params["sim_area"] is not None or sim_mask_in_raw_gaussian is not None:
                    pos = torch.cat([pos, unselected_pos], dim=0)
                    cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    shs = torch.cat([shs_render, unselected_shs], dim=0)
                    
                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                rendering, raddi = rasterize(
                    means3D=pos,
                    means2D=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D,
                )[:2]
                if frame==0:
                    hah_image_prompt=rendering.clone().detach()
                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                if render_force==True:
                    force_in_2d_scale = 80  # unit as pixel
                    for frc_inf in force_info:
                        start_time = frc_inf["start_frame"]
                        end_time = frc_inf["end_frame"]
                        if frame>=start_time and frame<end_time:
                            two_points = torch.stack([frc_inf["point"],
                                                      frc_inf["point"] +frc_inf["force"]], dim=0)
                            arrow_2d = render_arrow_in_screen(current_camera, two_points).cpu().detach().numpy()
                            start, vec_2d = arrow_2d[0], arrow_2d[1] - arrow_2d[0]
                            vec_2d = vec_2d / np.linalg.norm(vec_2d)

                            start = start  # + np.array([540.0, 288.0])

                            cv2_img = cv2.circle(
                                cv2_img, (int(start[0]), int(start[1])), 40, (255, 255, 255), 8
                            )

                            # draw arrow in img
                            end = start + vec_2d * force_in_2d_scale
                            end = end.astype(np.int32)
                            start = start.astype(np.int32)
                            cv2_img = cv2.arrowedLine(
                                cv2_img, (start[0], start[1]), (end[0], end[1]), (0, 255, 255), 8
                            )

                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                assert args.output_path is not None
                cv2.imwrite(
                    os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                    255 * cv2_img,
                )
            save_video(args.output_path, os.path.join(args.output_path, 'video%02d.mp4' % batch))
    return hah_image_prompt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--render_force", type=bool, default=False)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--method_type", type=str, default="MotionPhysics",
                        choices=["DreamPhysics", "PhysDreamer", "PhysFlow","MotionPhysics"],
                        help="Select the method type: MotionPhysics, Motion, or Physics.")
    parser.add_argument("--guidance_config", type=str, default="./config/guidance/ms_guidance.yaml")
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--total_batch", type=int, default=100)
    parser.add_argument("--frame_per_stage", type=int, default=16)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--downsample", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
        optimize_params
    ) = decode_param_json(args.physics_config)

    if args.downsample != 1.:
        for k in optimize_params["line"]:
            optimize_params["line"][k] *= args.downsample
        optimize_params["bbox_2d"] = [v*args.downsample for v in optimize_params["bbox_2d"]]
    if args.method_type == "PhysDreamer":
         material_params["material"] = "jelly"

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)

    


    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_screen_points = params["screen_points"]
    
    init_opacity = params["opacity"]
    init_cov = params["cov3D_precomp"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    clean_xyzs = None
    moving_pts = None

    clean_points_path = os.path.join(model_path, "clean_object_points.ply")
    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(clean_points_path) and os.path.exists(moving_pts_path):
        clean_xyzs = pcu.load_mesh_v(clean_points_path)
        clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")
        moving_pts_path = clean_points_path.replace("clean_object_points.ply",
                                                    "moving_part_points.ply")
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().cuda()
    elif os.path.exists(moving_pts_path):
        clean_xyzs = pcu.load_mesh_v(moving_pts_path)
        clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")

    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]

    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )

    
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )

    if moving_pts is not None:
        moving_rotated_pos = apply_rotations(moving_pts, rotation_matrices)
    rotated_pos = apply_rotations(init_pos, rotation_matrices)
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)




    sim_mask_in_raw_gaussian = None
    if preprocessing_params["sim_area"] is not None or clean_xyzs is not None:
        if clean_xyzs is not None:
            print('clean_xyzs.max: ',clean_xyzs.max()) # 0.9435
            print('clean_xyzs.min: ',clean_xyzs.min()) # -0.4408
            if "playdoh" in model_path:
                thres = 1.0/material_params["n_grid"]
            else:
                thres = 0.5/material_params["n_grid"]
            not_sim_maks = find_far_points(init_pos, clean_xyzs, thres=thres).bool()
            print('not_sim_mask: ',torch.count_nonzero(not_sim_maks)) # 515012,
            sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)

        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        if preprocessing_params["sim_area"] is not None:
            boundary = preprocessing_params["sim_area"]
            assert len(boundary) == 6
            for i in range(3):
                mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])
      
        if sim_mask_in_raw_gaussian is not None:
            mask = sim_mask_in_raw_gaussian
        
        unselected_pos = init_pos[~mask, :]
        unselected_cov =apply_inverse_cov_rotations(init_cov[~mask, :], rotation_matrices)
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos_complete = rotated_pos.detach().clone()
        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    scaling = 1.0
    if "biological_ball" in model_path:
        scaling = 0.6
    if 'cat' in model_path:
        scaling = 0.7
    if 'letter' in model_path:
        scaling = 2.0
    if 'cream' in model_path:
        scaling = 0.8
    if 'toothpaste' in model_path:
        scaling = 0.6
    if 'playdoh' in model_path:
        scaling = 0.75
    
    if material_params["grid_lim"]==1.0:
        # 0.1389, 0.6944
        pos_max = rotated_pos.max()
        pos_min = rotated_pos.min()
        scale_origin = 1/((pos_max - pos_min) * 1.8)
        original_mean_pos = -(-pos_min + (pos_max - pos_min) * 0.25)
        transformed_pos = (rotated_pos - original_mean_pos)*scale_origin
        shif_no = 0.0
    else:
        transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, scaling=scaling)
        shif_no = 1.0
    transformed_pos = shift2center111(transformed_pos,shif_no)
    if os.path.exists(clean_points_path):
        rotated_pos_complete = (rotated_pos_complete - original_mean_pos) * scale_origin
        rotated_pos_complete = shift2center111(rotated_pos_complete,shif_no)


    init_cov = scale_origin * scale_origin * init_cov

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]
    
    mpm_init_pos = transformed_pos.to(device=device)
    if "biological_ball" in model_path or "new_obj1_test" in model_path:
        mpm_init_pos = torch.load(model_path+"/ball_filled.pt")
    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)
    
    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:],shif_no), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print("gs.xyz", gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    
    mpm_solver = MPM_Simulator_WARP(10)

    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    freeze_pts = None
    freeze_mask = None
    if moving_pts is not None:


        transformed_moving_pos = (moving_rotated_pos - original_mean_pos) * scale_origin
        transformed_moving_pos = shift2center111(transformed_moving_pos,shif_no)

        freeze_mask = find_far_points(
            mpm_init_pos, 
            transformed_moving_pos, 
            thres=2 / material_params["n_grid"]
        ).bool()

        freeze_pts = mpm_init_pos[freeze_mask, :]

        freeze_mask_completed = find_far_points(
            rotated_pos_complete, 
            transformed_moving_pos, 
            thres=2 / material_params["n_grid"]
        ).bool()
        freeze_pts_complete = rotated_pos_complete[freeze_mask_completed, :]
        freeze_pts = torch.cat([freeze_pts,freeze_pts_complete],dim=0)
        print('freeze_pts.shape: ',freeze_pts.shape)
    
    sim_pos = apply_inverse_rotations(
        undotransform2origin(undoshift2center111(mpm_init_pos.clone(),shif_no), 
                             scale_origin, original_mean_pos),rotation_matrices,)
    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    
    force_info = set_boundary_conditions(mpm_solver, bc_params, time_params,freeze_pts,sim_pos)



    tape = wp.Tape()

    # mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )

    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_perframe = time_params["substep_perframe"]
    total_time = time_params["total_time"]
    frame_num = time_params["frame_num"]

    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    
    lr = {}
    stage_num = 8
    frame_per_stage = args.frame_per_stage

    for param_key in optimize_params['lr']:
        lr[param_key] = optimize_params['lr'][param_key][0]

    print('the prompt: ',optimize_params["prompt"])
    guidance_config = None
    if args.method_type == "MotionPhysics":
        from video_distillation_MotionPhysics.ms_guidance import ModelscopeGuidance
        from video_distillation_MotionPhysics.prompt_processors import ModelscopePromptProcessor
        from utils.headmlp import *
        from trainning_scripts.motionphysics_train import *
        guidance_config= args.guidance_config
        spatial_head = SpatialHead(num_channels=4, num_layers=2, kernel_size=1).cuda()
        target_spatial_head = SpatialHead(num_channels=4, num_layers=2, kernel_size=1).cuda()
        spatial_head.train()
        target_spatial_head.load_state_dict(spatial_head.state_dict())
        target_spatial_head.train()
        target_spatial_head.requires_grad_(False)
        optimizer= torch.optim.AdamW(
            spatial_head.parameters(),
            lr=2e-5,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1e-08)

    elif args.method_type == "PhysDreamer":
        from video_trainning_PhysDream.physdream_utils import *
        from trainning_scripts.physdream_train import *
        sim_aabb = torch.stack(
            [torch.min(mpm_init_pos, dim=0)[0], 
             torch.max(mpm_init_pos, dim=0)[0]], dim=0)
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        sim_fields = create_spatial_fields(24,output_dim=1,aabb=sim_aabb).cuda()
        sim_fields.train()
        velo_fields = create_spatial_fields(24, 3, sim_aabb, add_entropy=False).cuda()
        velo_fields.train()
        E_nu_list = init_trainable_params()
        for p in E_nu_list:
            p.requires_grad = True

        optim_list = [
            {"params": E_nu_list, "lr": 1e-3 * 1e-3},
            {"params": sim_fields.parameters(),
             "lr": 1e-3, "weight_decay": 1e-4,},
            {"params": velo_fields.parameters(),
             "lr": 1e-3 * 0.1,"weight_decay": 1e-4,}]
        optimizer = torch.optim.AdamW(
            optim_list,
            lr=1e-2,
            weight_decay=0.0,)
        guidance_path = os.path.join(args.model_path, 'images_generated')
        if not os.path.exists(guidance_path):
            AssertionError("Guidance frames do not exist!")
        guidance = prepare_gt_frames(guidance_path, downsample=args.downsample, num_frames=frame_per_stage)

    elif args.method_type == "DreamPhysics":
        guidance_config= args.guidance_config.replace("ms_guidance.yaml","ms_raw_guidance.yaml")
        from video_distillation_DreamPhysics.ms_guidance import ModelscopeGuidance
        from trainning_scripts.dreamphysics_train import *
        from video_distillation_DreamPhysics.prompt_processors import ModelscopePromptProcessor
    elif args.method_type == "PhysFlow":
        from video_distillation_PhysFlow.cogv_guidance import CogVideoGuidance
        from trainning_scripts.physflow_train import *
        guidance_path = os.path.join(args.model_path, 'images_generated')
        if not os.path.exists(guidance_path):
            AssertionError("Guidance frames do not exist!")
        guidance = CogVideoGuidance(guidance_path, downsample=args.downsample, num_frames=frame_per_stage)
    if args.method_type in ["MotionPhysics","DreamPhysics"]:
        yaml_confs = OmegaConf.load(guidance_config)
        yaml_confs.prompt_processor.prompt = optimize_params["prompt"]
        guidance = ModelscopeGuidance(yaml_confs.guidance)
        prompt_processor = ModelscopePromptProcessor(yaml_confs.prompt_processor)
        prompt_utils = prompt_processor()

    substep_dt = total_time / (frame_num*substep_perframe)
    image_prompt = None
    for batch in range(args.total_batch):
        print('processing_batch: ',batch)
        noise_shs = 0
        noise_pos = 0
        if args.method_type == "PhysDreamer":
            mpm_init_v,youngs_modulus,poisson_ratio = prepared_E_nu_velo(E_nu_list,
                                                                         mpm_solver,
                                                                         mpm_init_pos,
                                                                         freeze_mask,velo_fields,sim_fields,device)
            
        single_image_prompt = evaluate_mp4(mpm_solver,mpm_init_pos,mpm_init_vol,mpm_init_cov,
                    batch,stage_num,frame_per_stage,camera_params,gaussians,
                    pipeline,background,height,width,args,force_info,args.render_force)
        if single_image_prompt!=None:
            image_prompt = single_image_prompt
        mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol) 
        gc.collect()
        torch.cuda.empty_cache()  
        loss_value = 0.
        img_list = []
        tape.reset()
        if mpm_solver.mpm_model.material not in [5,7]:
            with tape:
                mpm_solver.finalize_mu_lam()
        
        for frame in range(substep_perframe * (batch % stage_num)):
            mpm_solver.p2g2p(substep_dt, device=device)
        print('each_frame_process_step: ',substep_perframe * (1 + stage_num) )
        for frame in tqdm(range(frame_per_stage)):
            delta_r = camera_params["delta_r"]
            if 'alocasia' in args.model_path:
                delta_r = camera_params["delta_r"] if frame < (frame_num+1)/2 else camera_params["delta_r"] / frame * ((stage_num * frame_per_stage)-frame) 
            current_camera,_ = get_camera_view(
                model_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"],
                init_elevation=camera_params["init_elevation"],
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                current_frame=frame,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=delta_r,
                downsample=args.downsample
            )
            rasterize = initialize_resterize(
                current_camera, gaussians, pipeline, background
            )
            for _ in range(substep_perframe * (1 + stage_num) - 1):
                mpm_solver.p2g2p(substep_dt, device=device)
            with tape:
                 mpm_solver.p2g2p(substep_dt, device=device)

            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = mpm_solver.export_particle_cov_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)
            
            pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos,shif_no), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,)

            cov3D = init_cov / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None or sim_mask_in_raw_gaussian is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            if frame==0 and args.method_type == "MotionPhysics":
                epsilon = 1e-1  
                noise_shs = 2*epsilon * torch.randn_like(shs)  # leaf tensor, requires_grad=False 
                noise_pos = 1*epsilon * torch.randn_like(pos)

            random_shs = shs + noise_shs
            random_pos = pos + noise_pos 
            
            colors_precomp = convert_SH(random_shs, current_camera, gaussians, random_pos, rot)
           
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp= colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )[:2]
            img_list.append(rendering)
        img_list = torch.stack(img_list)
        if args.method_type == "MotionPhysics":
            training_scripts(guidance, img_list, prompt_utils, camera_params,
                            frame_per_stage, spatial_head, target_spatial_head,
                            optimizer, update_ema, mpm_solver, optimize_params,device,lr,tape,batch,stage_num)
        elif args.method_type == "DreamPhysics":
            training_scripts(guidance, img_list, prompt_utils, camera_params, 
                            frame_per_stage,mpm_solver, optimize_params,device,lr,tape,batch,stage_num)
        elif args.method_type == "PhysFlow":
            training_scripts(guidance, img_list,image_prompt, camera_params, 
                            frame_per_stage,mpm_solver, optimize_params,device,lr,tape,batch,stage_num)
        elif args.method_type == "PhysDreamer":
            trainning_scripts(tape,mpm_solver,velo_fields,
                                sim_fields,guidance,img_list,device,optimizer,
                                mpm_init_v,youngs_modulus,poisson_ratio)
        else:   
            print('not implementated')
