from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3, wp_clamp, update_param
from utils.update_grad import *
import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
import os
SCALE_E = 1e7

def save_images_with_alpha(img_list, batch, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i, img_tensor in enumerate(img_list):
   
        img = to_pil_image(img_tensor)

        img_rgba = img.convert("RGBA")
        datas = img_rgba.getdata()

        new_data = []
        for item in datas:
            if item[0] < 10 and item[1] < 10 and item[2] < 10:
                new_data.append((0, 0, 0, 0))  
            else:
                new_data.append(item[:3] + (255,)) 

        img_rgba.putdata(new_data)

        # 保存文件
        out_path = os.path.join(save_dir, f"{batch}_rendering_{i:03d}.png")
        img_rgba.save(out_path)
        print(f"Saved: {out_path}")

def training_scripts(guidance, img_list, prompt_utils, camera_params, frame_per_stage, spatial_head, target_spatial_head,
                     optimizer, update_ema, mpm_solver, optimize_params,device,lr,tape,batch,stage_num):
    latents, latents_predicted = guidance(
        img_list, prompt_utils,
        torch.Tensor([camera_params['init_elevation']]),
        torch.Tensor([camera_params['init_azimuthm']]),
        torch.Tensor([camera_params['init_radius']]),
        rgb_as_latents=False,
        num_frames=frame_per_stage,
        train_dynamic_camera=True
    )



    pred_cd = spatial_head(latents.float())

    with torch.no_grad():
        target_cd = target_spatial_head(latents_predicted.detach().float())
    loss = torch.sum(
        torch.sqrt((pred_cd - target_cd).float() ** 2 + 0.001**2) - 0.001
    )
    loss = loss / stage_num
    #print('total_loss: ',loss)
    loss.backward(retain_graph=True)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    update_ema(target_spatial_head.parameters(), spatial_head.parameters(), 0.95)

    grad_x = mpm_solver.mpm_state.particle_x.grad
    grad_cov = mpm_solver.mpm_state.particle_cov.grad
    grad_r = mpm_solver.mpm_state.particle_R.grad

    loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    wp.launch(sum_vec3, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_x, grad_x], [loss_wp], device=device)
    wp.launch(sum_array, mpm_solver.n_particles * 6, [mpm_solver.mpm_state.particle_cov, grad_cov], [loss_wp], device=device)
    wp.launch(sum_mat33, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_R, grad_r], [loss_wp], device=device)

    tape.backward(loss=loss_wp)

    # Clean up
    del grad_x, grad_r, grad_cov, img_list, loss
    torch.cuda.empty_cache()

    # ================= Material-specific Gradient Updates =================
    material = mpm_solver.mpm_model.material

    if material == 0:  # Elastic
        update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['E'], lower=-4.0, upper=-0.1, log_name="E", scale=1, gn=True)
        update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['nu'], lower=-1.3, upper=-0.4, log_name="nu", scale=1, gn=True)

    elif material in [1, 4]:  # Metal or Plasticine
        update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, n_particles=mpm_solver.n_particles,scale=1, gn=True,
                          lrate=lr['E'], lower=-4.0, upper=(-0.4 if material == 4 else 0.5), log_name="E")
        update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['nu'], lower=-4.0, upper=-0.4, log_name="nu", scale=1,gn=True)
        update_grad_param(mpm_solver.mpm_model.yield_stress, mpm_solver.mpm_model.yield_stress.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['yield_stress'], lower=-4.0, upper=(-1.0 if material == 4 else 0.0), log_name="yield_stress",scale=1,gn=True)

    elif material == 2:  # Sand
        update_grad_param(mpm_solver.mpm_model.friction_angle, mpm_solver.mpm_model.friction_angle.grad, n_particles=1,
                          lrate=lr['friction_angle'], lower=0.0, upper=2.0, log_name="friction_angle", scale=1,gn=False)

    elif material == 7:  # Non-Newtonian
        update_grad_param(mpm_solver.mpm_model.bulk, mpm_solver.mpm_model.bulk.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['bulk'], lower=-4, upper=3, log_name="bulk", scale=1,gn=True)

    elif material == 3:  # Foam
        for param_name, lower, upper in [('E', -4.0, -0.4), ('nu', -4.0, -0.4), ('yield_stress', -4.0, -0.8)]:
            update_grad_param(getattr(mpm_solver.mpm_model, param_name),
                              getattr(mpm_solver.mpm_model, param_name).grad,
                              n_particles=mpm_solver.n_particles,
                              lrate=lr[param_name], lower=lower, upper=upper,
                              log_name=param_name, scale=1,gn=True)
        update_grad_param(mpm_solver.mpm_model.plastic_viscosity, mpm_solver.mpm_model.plastic_viscosity.grad, 1,
                          lrate=lr['plastic_viscosity'], lower=-4.0, upper=-1.0, log_name="plastic_viscosity",scale=1,gn=True)

    elif material == 6:  # Complex Fluid
        if batch < 50:
            update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, n_particles=mpm_solver.n_particles,
                              lrate=lr['E'], lower=-7.0, upper=-0.4, log_name="E", scale=1,gn=True)
            update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, n_particles=mpm_solver.n_particles,
                              lrate=lr['nu'], lower=-4.0, upper=-0.31, log_name="nu", scale=1,gn=True)
        else:
            update_grad_param(mpm_solver.mpm_model.yield_stress, mpm_solver.mpm_model.yield_stress.grad, n_particles=mpm_solver.n_particles,
                              lrate=lr['yield_stress'], lower=-4.0, upper=-0.8, log_name="yield_stress", scale=1)
            update_grad_param(mpm_solver.mpm_model.plastic_viscosity, mpm_solver.mpm_model.plastic_viscosity.grad, 1,
                              lrate=lr['plastic_viscosity'], lower=-4.0, upper=-1.0, log_name="plastic_viscosity", scale=1,gn=True)

        nu = wp.to_torch(mpm_solver.mpm_model.nu)
        E = wp.to_torch(mpm_solver.mpm_model.E)

        fluid_viscosity = SCALE_E * E / (2. * (1. + nu))
        bulk = SCALE_E * E / (3. * torch.clamp(1. - 2. * nu, min=1e-4))

        print(f"   --> fluid_viscosity: {torch.mean(fluid_viscosity).item()}")
        print(f"   --> bulk: {torch.mean(bulk).item()}")

    elif material == 5:  # Newtonian Fluid
        update_grad_param(mpm_solver.mpm_model.mu, mpm_solver.mpm_model.mu.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['mu'], lower=-7.3, upper=-0.699, log_name="mu",scale=mpm_solver.n_particles,gn=True)
        update_grad_param(mpm_solver.mpm_model.bulk, mpm_solver.mpm_model.bulk.grad, n_particles=mpm_solver.n_particles,
                          lrate=lr['bulk'], lower=-4, upper=3, log_name="bulk", scale=mpm_solver.n_particles,gn=True)

    # =================== Learning Rate Scheduling =========================
    for param_key in optimize_params['lr']:
        param_lr = optimize_params['lr'][param_key]
        warmup = 0 if len(param_lr) < 4 else param_lr[3]

        if material == 6:
            max_steps = None if len(param_lr) < 3 else param_lr[2]//2
            if batch < 50:
                if param_key in ['E', 'nu']:
                    lr[param_key] = lr_scheduler(param_lr[0], param_lr[1], batch, 50, warmup_steps=warmup, max_steps=max_steps)
            else:
                if param_key in ['yield_stress', 'plastic_viscosity']:
                    lr[param_key] = lr_scheduler(param_lr[0], param_lr[1], batch-50, 50, warmup_steps=warmup, max_steps=max_steps)
        else:
            max_steps = None if len(param_lr) < 3 else param_lr[2]
            lr[param_key] = lr_scheduler(param_lr[0], param_lr[1], batch, 100, warmup_steps=warmup, max_steps=max_steps)

    # =================== Logging =========================
    logs = {}
    for param_key in optimize_params['lr']:
        logs[f'param/{param_key}_max'] = wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).max().item()
        logs[f'param/{param_key}_min'] = wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).min().item()
        if param_key in lr:
            logs[f'lr/{param_key}'] = lr[param_key]

    print('logs:', logs)
