from diffusers import IFPipeline, IFSuperResolutionPipeline
from diffusers import DiffusionPipeline
import torch
import pdb 
import numpy as np 
from PIL import Image
import os
from utils import set_scheduler, seed_everything, register_attention_control
from diffusers.utils import pt_to_pil

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.deepfloyd_if import IFPipelineOutput
import PIL

from utils import AttentionStore, get_mask
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gaussian_smoothing import GaussianSmoothing
from diffusers import DDIMScheduler
from diffusers.utils import randn_tensor

class MyIFSuperPipeline(IFSuperResolutionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = None,
        width: int = None,
        image: Union[PIL.Image.Image, np.ndarray, torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 250,
        clean_caption: bool = True,
        attention_store: AttentionStore = None, 
        use_rate=True,
        R=4,
    ):
        
        # 1. Check inputs. Raise error if not correct

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.check_inputs(
            prompt,
            image,
            batch_size,
            noise_level,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters

        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        num_channels = self.unet.config.in_channels // 2
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            num_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare upscaled image and noise level
        image = self.preprocess_image(image, num_images_per_prompt, device)
        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
        noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
        upscaled = self.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

        if do_classifier_free_guidance:
            noise_level = torch.cat([noise_level] * 2)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([intermediate_images, upscaled], dim=1)

                model_input = torch.cat([model_input] * 2) if do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
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
                        # print(rate.min())
                        
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

                    noise_pred = noise_pred_uncond + guidance_scale*rate* (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
                    
                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 9. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 10. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 11. Convert to PIL
            image = self.numpy_to_pil(image)

            # 12. Apply watermark
            # if self.watermarker is not None:
            #     self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pt":
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            # 9. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 10. Run safety checker
            image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, nsfw_detected, watermark_detected)

        return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)



class MyIFPipeline(IFPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_store: AttentionStore = None, 
        use_rate=True,
        R=4
    ):
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

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
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )
       
        # pdb.set_trace()

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        # 5. Prepare intermediate images
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # HACK: see comment in `enable_model_cpu_offload`
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = (
                    torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
                )
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    if use_rate:
                        ca_mask, fore_mask = get_mask(attention_store, r=R)
                  
                    if use_rate:
                        mask_t  =F.interpolate(ca_mask, scale_factor=R, mode = 'nearest')

                        ###eps
                        model_delta = (noise_pred_text - noise_pred_uncond)
                        model_delta_norm = model_delta.norm(dim=1, keepdim=True) # b 1 64 64

                        delta_mask_norms = (model_delta_norm*mask_t).sum([2,3])/(mask_t.sum([2,3])+1e-8) # b 77
                        upnormmax = delta_mask_norms.max(dim=1)[0] # b
                        upnormmax = upnormmax.unsqueeze(-1)
                        mean_norms = model_delta_norm.mean([2,3])

                        tmp_mask = (mask_t.sum([2,3])>0).float()
                        
                        up=mean_norms
                        down = delta_mask_norms
                        
                        rate = up*(tmp_mask)/(down+1e-8) # b 257
                        rate = (rate.unsqueeze(-1).unsqueeze(-1)*mask_t).sum(dim=1, keepdim=True) # b 1, 64 64
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

                    noise_pred = noise_pred_uncond + guidance_scale*rate* (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
                elif isinstance(self.scheduler, DDIMScheduler):
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, intermediate_images)

        image = intermediate_images

        if output_type == "pil":
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            # image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)

            # 11. Apply watermark
            # if self.watermarker is not None:
            #     image = self.watermarker.apply_watermark(image, self.unet.config.sample_size)
        elif output_type == "pt":
            nsfw_detected = None
            watermark_detected = None

            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        else:
            # 8. Post-processing
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            # 9. Run safety checker
            # image, nsfw_detected, watermark_detected = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, nsfw_detected, watermark_detected)

        return IFPipelineOutput(images=image, nsfw_detected=nsfw_detected, watermark_detected=watermark_detected)
