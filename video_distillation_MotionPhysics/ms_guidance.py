## This implementation is based on the threestudio extension Animate124: https://github.com/HeliosZhao/Animate124/tree/threestudio 

from jaxtyping import Float, Int

import torch
import torch.nn.functional as F
from torch import Tensor
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
#from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

from video_distillation.prompt_processors import PromptProcessorOutput
from utils.threestudio_utils import parse_version, cleanup, get_device, C


class ModelscopeGuidance:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = CogVideoXPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                print(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.transformer.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.transformer = self.pipe.transformer.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        num_inference_steps = getattr(self.cfg, "num_inference_steps", 50)
        self.scheduler.set_timesteps(num_inference_steps)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        ).to(torch.float32)

        self.grad_clip_val = None

        # Extra for latents
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        ## set spatial size
        self.spatial_size = (256, 256)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_transformer(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype


        return self.transformer(
            hidden_states = latents.to(self.weights_dtype),
            timestep = t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 N 320 576"], normalize: bool = True
    ) -> Float[Tensor, "B 4 40 72"]:
        if len(imgs.shape) == 4:
            print("Only given an image an not video")
            imgs = imgs[:, :, None]
        
        batch_size, channels, num_frames, height, width = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        input_dtype = imgs.dtype
        if normalize:
            imgs = imgs * 2.0 - 1.0

        if self.cfg.low_ram_vae > 0:
            vnum = self.cfg.low_ram_vae
            mask_vae = torch.randperm(imgs.shape[0]) < vnum
            with torch.no_grad():
                posterior_mask = torch.cat(
                    [
                        self.vae.encode(
                            imgs[~mask_vae][i : i + 1].to(self.weights_dtype)
                        ).latent_dist.sample()
                        for i in range(imgs.shape[0] - vnum)
                    ],
                    dim=0,
                )
            posterior = torch.cat(
                [
                    self.vae.encode(
                        imgs[mask_vae][i : i + 1].to(self.weights_dtype)
                    ).latent_dist.sample()
                    for i in range(vnum)
                ],
                dim=0,
            )
            posterior_full = torch.zeros(
                imgs.shape[0],
                *posterior.shape[1:],
                device=posterior.device,
                dtype=posterior.dtype,
            )
            posterior_full[~mask_vae] = posterior_mask
            posterior_full[mask_vae] = posterior
            latents = posterior_full * self.vae.config.scaling_factor
        else:
            posterior = self.vae.encode(imgs.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.weights_dtype)).latent_dist
            latents = (posterior.sample().transpose(1, 2) * self.vae.config.scaling_factor)
        
        """        
        latents = (
            latents[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + latents.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        """
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        # TODO: Make decoding align with previous version
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            
            noise_pred = self.forward_transformer(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        scheduler_out = self.scheduler.step(noise_pred, t, latents_noisy)
        latents_denoised = scheduler_out.pred_original_sample
        return latents_denoised

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents: bool = False,
        num_frames: int = 12,
        train_dynamic_camera: bool = False,
    ):
        rgb_BCHW = rgb#.permute(0, 3, 1, 2)
        batch_size = rgb_BCHW.shape[0] // num_frames
        latents: Float[Tensor, "B 4 40 72"]
        if train_dynamic_camera:
            elevation = elevation[[0]]
            azimuth = azimuth[[0]]
            camera_distances = camera_distances[[0]]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (self.spatial_size[0]//8, self.spatial_size[1]//8), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, self.spatial_size, mode="bilinear", align_corners=False
            )
            rgb_BCHW_512 = rgb_BCHW_512.permute(1, 0, 2, 3)[None] # 1,4,B,H,W
            latents = self.encode_images(rgb_BCHW_512) # encode the images
            
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        latents_denoised = self.compute_grad_sds(latents, text_embeddings, t)
        return latents,latents_denoised
        """
        grad = self.compute_grad_sds(latents, text_embeddings, t)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        print('latentshape: ',latents.shape)
        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return {
            "loss_sds_video": loss_sds,
            # "grad_norm": grad.norm(),
        }
        """

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        # t annealing from ProlificDreamer
        if (
            self.cfg.anneal_start_step is not None
            and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )
