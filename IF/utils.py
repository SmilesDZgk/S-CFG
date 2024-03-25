from diffusers import DiffusionPipeline
import torch
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_scheduler(pipe:DiffusionPipeline, scheduler):
    if scheduler=='DDIM':
        pass
        # from diffusers import DDIMScheduler
        # DDIMScheduler
        # pipe.scheduler = DDIMScheduler.from_config(
        #     pipe.scheduler.config, ) #https://arxiv.org/pdf/2305.08891.pdf
        #rescale_betas_zero_snr=False,  variance_type = 'fixed_small', set_alpha_to_one=False
        # pipe.scheduler.variance_type = 'fixed_small'
        
    elif scheduler=='PNDM':
        from diffusers import PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(
            pipe.scheduler.config,  timestep_spacing="leading") #https://arxiv.org/pdf/2305.08891.pdf
    
    elif scheduler=='DPMSlover':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,  solver_order=2, algorithm_type='dpmsolver++' ) 
        #solver_order=2 for conditional 3 for unconditional sampling 
    
    elif scheduler=='Euler':
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config ) 
        #solver_order=2 for conditional 3 for unconditional sampling 
    else:
        return pipe
    return pipe




from diffusers.models.cross_attention import CrossAttention
import abc
from typing import Union, Tuple, List
from einops import rearrange
class MyCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):

        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)


        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(rearrange(attention_probs, '(b h) s t -> b h s t', h=attn.heads).mean(1), self.place_in_unet)


        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states
    

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = MyCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [], "mid_self": [], "up_self": []}
        return {"r2_cross": [],"r4_cross": [], "r8_cross": [], "r16_cross": [],
                "r2_self": [], "r4_self": [], "r8_self": [], "r16_self": []}

    def forward(self, attn, place_in_unet: str):  ####TODO 修改key名称
        h = int(attn.size(1)**(0.5))
        r = int(self.H/h)
        # print(r)
        key_cross = f"r{r}_cross"
        key_self = f"r{r}_self"
        cross_attn, self_attn = (attn+1e-20).split([attn.size(-1)-attn.size(-2), attn.size(-2)],dim=-1)
        
        # if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        if r >= 2:  # avoid memory overhead
            self.step_store[key_cross].append(cross_attn)
            self.step_store[key_self].append(self_attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self,H, save_global_store=False, name='stage1'):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.H = H
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
        self.name=name
        # self.count=0

import torch.nn.functional as F
from gaussian_smoothing import GaussianSmoothing

def get_mask(attention_store: AttentionStore,r: int=4):
    """ Aggregates the attention across the different layers and heads at the specified resolution. """

    key_corss = f"r{r}_cross"
    key_self = f"r{r}_self"
    curr_r = r
    r_r = 1
    new_ca = 0
    new_fore=0
    a_n=0
    attention_maps = attention_store.get_average_attention()
    while curr_r<=8:
        key_corss = f"r{curr_r}_cross"
        key_self = f"r{curr_r}_self"
        if len(attention_maps[key_self])==0:
            curr_r = int(curr_r*2)
            r_r*=2
            continue
        
        sa = torch.stack(attention_maps[key_self][-3:], dim=1)
        ca = torch.stack(attention_maps[key_corss][-3:], dim=1)
        attn_num = sa.size(1)
        sa = rearrange(sa, 'b n h w -> (b n) h w')
        ca = rearrange(ca, 'b n h w -> (b n) h w')

        curr = 0 # b hw c=hw
        curr +=sa
        ssgc_sa = curr
        ssgc_n =4
        for _ in range(ssgc_n-1):
            curr = sa@sa
            ssgc_sa += curr
        ssgc_sa/=ssgc_n
        sa = ssgc_sa
        ########smoothing ca
        ca = sa@ca # b hw c

        h=w = int(sa.size(1)**(0.5))

        ca = rearrange(ca, 'b (h w) c -> b c h w', h=h )
        if r_r>1:
            ca = F.interpolate(ca, scale_factor=r_r, mode="bilinear") # b 77 32 32

        #####Gaussian Smoothing
        kernel_size = 3
        sigma = 0.5
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(ca.device)
        channel = ca.size(1)
        ca= rearrange(ca, ' b c h w -> (b c) h w' ).unsqueeze(1)
        ca = F.pad(ca, (1, 1, 1, 1), mode='reflect')
        ca = smoothing(ca.float()).squeeze(1)
        ca = rearrange(ca, ' (b c) h w -> b c h w' , c= channel)


        ca_norm = ca/(ca.mean(dim=[2,3], keepdim=True)+1e-8) ### spatial  normlization 
        
        new_ca+=rearrange(ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1) 


        fore_ca = torch.stack([ca[:,0],ca[:,1:].sum(dim=1)], dim=1)
        froe_ca_norm = fore_ca/fore_ca.mean(dim=[2,3], keepdim=True) ### spatial  normlization 
        new_fore += rearrange(froe_ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1) 
        
        a_n+=attn_num

        curr_r = int(curr_r*2)
        r_r*=2

        

        # # for sa, ca in zip(attention_maps[key_self][-3:], attention_maps[key_corss][-3:]):
        # #     curr = 0 # b hw c=hw
        # #     curr +=sa
        # #     ssgc_sa = curr
        # #     ssgc_n =4
        # #     for _ in range(ssgc_n-1):
        # #         curr = sa@sa
        # #         ssgc_sa += curr
        # #     ssgc_sa/=ssgc_n
        # #     sa = ssgc_sa
        # #     ########smoothing ca
        # #     ca = sa@ca # b hw c

        # #     h=w = int(sa.size(1)**(0.5))

        # #     ca = rearrange(ca, 'b (h w) c -> b c h w', h=h )
        # #     sa = rearrange(sa, 'b (h w) c -> b c h w', h=h )
        # #     if r_r>1:
        # #         ca = F.interpolate(ca, scale_factor=r_r, mode="bilinear") # b 77 32 32
        # #         sa = F.interpolate(sa, scale_factor=r_r, mode="bilinear") # b hw 32 32
        # #         sa = rearrange(sa, 'b (s t) h w -> b (h w) s t', s=h )
        # #         sa = F.interpolate(sa, scale_factor=r_r, mode="bilinear") # b 32*32 32 32
        # #         sa = rearrange(sa, 'b (h w) s t -> b (s t) h w', h=r_r*h ) # b 32*32 32 32
        # #         sa /=sa.sum(1, keepdim=True)

            
        # #     new_sa+=sa


        # #     #####Gaussian Smoothing
        # #     kernel_size = 3
        # #     sigma = 0.5
        # #     smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(ca.device)
        # #     channel = ca.size(1)
        # #     ca= rearrange(ca, ' b c h w -> (b c) h w' ).unsqueeze(1)
        # #     ca = F.pad(ca, (1, 1, 1, 1), mode='reflect')
        # #     ca = smoothing(ca.float()).squeeze(1)
        # #     ca = rearrange(ca, ' (b c) h w -> b c h w' , c= channel)

        # #     new_ca+=ca/(ca.mean(dim=[2,3], keepdim=True)+1e-8) ### spatial  normlization 

        # #     fore_ca = torch.stack([ca[:,0],ca[:,1:].sum(dim=1)], dim=1)
        # #     new_fore += fore_ca/fore_ca.mean(dim=[2,3], keepdim=True) ### spatial  normlization 
        # #     a_n+=1

            
        # #     k+=1

        # curr_r = int(curr_r*2)
        # r_r*=2

    new_ca = new_ca/a_n
    new_fore = new_fore/a_n
    _,new_ca   = new_ca.chunk(2, dim=0) 
    fore_ca, _ = new_fore.chunk(2, dim=0)


    max_ca, inds = torch.max(new_ca[:,:], dim=1) 
    max_ca = max_ca.unsqueeze(1) # 
    ca_mask = (new_ca==max_ca).float() # b 77/10 16 16 
    
    max_fore, inds = torch.max(fore_ca[:,:], dim=1) 
    max_fore = max_fore.unsqueeze(1) # 
    fore_mask = (fore_ca==max_fore).float() # b 77/10 16 16 
    fore_mask = 1.0-fore_mask[:,:1] # b 1 16 16
    



    

    return  ca_mask, fore_mask 








