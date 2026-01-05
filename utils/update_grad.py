import torch
import numpy as np
import warp as wp
from mpm_solver_warp.mpm_utils import update_param
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
def save_latents(video_data,path="selected_frames.png"):
   
    video_data = video_data.squeeze(0)  

    selected_frame_indices = range(16)
    
    os.makedirs(path, exist_ok=True)

    for frame_idx in selected_frame_indices:
        frame_data = video_data[:, frame_idx, :, :]
        flattened_data = frame_data.reshape(len(frame_data), -1).T
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(flattened_data)
        rgb_frame = reduced_data.reshape(32, 32, 3)
        rgb_frame = (rgb_frame - rgb_frame.min()) / (rgb_frame.max() - rgb_frame.min())
        out_path = os.path.join(path, f"frame_{frame_idx:03d}.png")
        plt.imsave(out_path, rgb_frame)

def update_grad_param(param, param_grad, n_particles, lrate=1.0, lower=-1.0, upper=-0.4, gn=True, scale=1., log_name=None, debug=False):
    grad = wp.to_torch(param_grad) / scale
    print('grad_before: ',grad.max())
    if gn:
        max_grad, min_grad = torch.max(grad), torch.min(grad)
        range_grad = max_grad - min_grad
        grad = torch.where(
            range_grad > 1e-6, 
            (grad - min_grad) / range_grad - 0.5,
            torch.zeros_like(grad))
        print('grad_after: ',grad.max())
    if scale!=1: # PhysFlow
        grad_mean = torch.mean(grad)
        grad = torch.full_like(grad, grad_mean)

    if not debug:
        wp.launch(update_param, n_particles, [param, wp.from_torch(grad), lrate, upper, lower])

    if log_name is not None:
        print(f"- {log_name}: {torch.mean(wp.to_torch(param)).item()}, grad_{log_name}: {torch.mean(grad).item()}, lr_{log_name}: {lrate}")


def lr_scheduler(lr_init, lr_end, step, total_steps, warmup_steps=0, max_steps=None):
    if max_steps is None:
        max_steps = total_steps
    if step < warmup_steps:
        lr = float(step) / float(warmup_steps) * (lr_init - lr_end) + lr_end
    elif step < max_steps:
        lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos((step - warmup_steps) / (max_steps - warmup_steps) * np.pi))
    else:
        lr = lr_end
    return lr