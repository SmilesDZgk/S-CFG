from diffusers import StableDiffusionPipeline
import torch
import pdb 
import numpy as np 
from PIL import Image
import os
from utils import set_scheduler, seed_everything, register_attention_control

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from utils import AttentionStore, get_mask
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gaussian_smoothing import GaussianSmoothing
class MySDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        attention_store: AttentionStore = None, 
        use_rate=True,
        R=4,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if use_rate:
                        ca_mask, fore_mask = get_mask(attention_store, r=R)

                    if use_rate:
                        mask_t  =F.interpolate(ca_mask, scale_factor=R, mode = 'nearest')
                        mask_fore  =F.interpolate(fore_mask, scale_factor=R, mode = 'nearest')
                        

                        ###eps
                        model_delta = (noise_pred_text - noise_pred_uncond)
                        model_delta_norm = model_delta.norm(dim=1, keepdim=True) # b 1 64 64

                        delta_mask_norms = (model_delta_norm*mask_t).sum([2,3])/(mask_t.sum([2,3])+1e-8) # b 77
                        upnormmax = delta_mask_norms.max(dim=1)[0] # b
                        upnormmax = upnormmax.unsqueeze(-1)

                        fore_norms = (model_delta_norm*mask_fore).sum([2,3])/(mask_fore.sum([2,3])+1e-8) # b 1

                        

                        up = fore_norms
                        down = delta_mask_norms
                        

                        tmp_mask = (mask_t.sum([2,3])>0).float()
                        rate = up*(tmp_mask)/(down+1e-8) # b 257
                        rate = (rate.unsqueeze(-1).unsqueeze(-1)*mask_t).sum(dim=1, keepdim=True) # b 1, 64 64
                        
                        rate = torch.clamp(rate,min=0.8, max=3.0)
                        rate = torch.clamp_max(rate, 15.0/guidance_scale)


                        ###Gaussian Smoothing 
                        kernel_size = 3
                        sigma=0.5
                        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(rate.device)
                        rate = F.pad(rate, (1, 1, 1, 1), mode='reflect')
                        rate = smoothing(rate)


                        rate = rate.to(noise_pred_text.dtype)

                    else:
                        rate=1.0

                    noise_pred = noise_pred_uncond + guidance_scale *rate* (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)




if __name__=='__main__':
    path = 'path/to/stable-diffusion-v1-5-or-stable-diffusion-2-1'
    model_id = path
    pipe = MySDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    controller = AttentionStore(H = pipe.unet.config.sample_size)
    register_attention_control(pipe, controller)

    sample_dir = 'outputs/samples/'
    grid_count=0
    while True:
        start = 0
        batch_size=1
        cfg = 7.5
        steps = 50
        seed = 42
        use_rate=False # True for S-CFG, False for CFG
        scheduler = 'DPMSlover' # 'DPMSlover', 'DDIM' 
        
        seed_everything(seed=seed)
        set_scheduler(pipe, scheduler)
        prompts = ['A boy in red shirt']*batch_size # prompt
        # start+=batch_size
        images = pipe(prompts, num_inference_steps=steps,guidance_scale=cfg, eta=0.0, 
                    attention_store=controller, use_rate=use_rate).images 
        grid = np.hstack(images)
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_dir, f'grid-{grid_count:04}.png'))
        grid_count+=1