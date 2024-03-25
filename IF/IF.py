from IF_utils import MyIFPipeline, MyIFSuperPipeline
import torch
import pdb 
import numpy as np 
from PIL import Image
import os
from utils import set_scheduler, seed_everything, register_attention_control
from diffusers.utils import pt_to_pil


from utils import AttentionStore

import time
if __name__=='__main__':

    path1 = 'path/to/ IF-I-M-v1.0'
    path2 = 'path/to/IF-II-M-v1.0'

    stage_1 = MyIFPipeline.from_pretrained(path1, variant="fp16", torch_dtype=torch.float16)
    stage_2 = MyIFSuperPipeline.from_pretrained(path2, variant="fp16", text_encoder=None, torch_dtype=torch.float16)

    stage_1 = stage_1.to("cuda")
    stage_2 = stage_2.to("cuda")
    controller1 = AttentionStore(H = stage_1.unet.config.sample_size,name='stage1')
    register_attention_control(stage_1, controller1)

    controller2 = AttentionStore(H = stage_2.unet.config.sample_size, name='stage2')
    register_attention_control(stage_2, controller2)

    sample_dir = 'outputs/samples/'
    grid_count=0
    while True:
        start = 0
        batch_size=1
        cfg = 7.0 #7.5
        steps = 50
        seed = 42
        use_rate=False # True for S-CFG, False for CFG
        scheduler = 'DPMSlover' #'DPMSlover', 'DDIM'
        eta=0.0
        pdb.set_trace()
        seed_everything(seed=seed)
        generator = torch.manual_seed(seed)
        stage_1 = set_scheduler(stage_1, scheduler)
        stage_2 = set_scheduler(stage_2, scheduler)
        prompts = ['A brown hamster standing on a hair brush']*batch_size
        negprompts = ['']*len(prompts)

        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompts, negative_prompt=negprompts)
        images = stage_1(
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, 
                    generator=generator, output_type="pt", num_inference_steps=steps,guidance_scale=cfg,
                    attention_store=controller1, use_rate=use_rate, eta=eta
                ).images


        images = stage_2(
                    image=images,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds, num_inference_steps=50,guidance_scale=cfg,
                    generator=generator, #output_type="pt",
                    attention_store=controller2, use_rate=use_rate, eta=eta
                ).images
        grid = np.hstack(images)
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_dir, f'grid-{grid_count:04}.png'))
        grid_count+=1