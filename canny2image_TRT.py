from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from trt.clip_trt import CLIPTrt
from trt.vae_trt import VAETrt
from trt.ddim_trt import DDIMTrt

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        del self.model.cond_stage_model
        del self.model.control_model
        # del self.model.decode_first_stage
        torch.cuda.empty_cache()

        # self.ddim_sampler = DDIMSampler(self.model)
        # self.ddim_sampler.make_schedule(20)
        # self.ddim_sampler.init_trt()
        self.clip = CLIPTrt()
        self.vae = VAETrt()
        self.ddim = DDIMTrt(self.model)


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.unsqueeze(control, 0)
            # control = torch.stack([control for _ in range(num_samples)], dim=0)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            img = torch.randn((1, 4, 32, 48), device=control.device)

            context, hint = self.clip.run(prompt, a_prompt, n_prompt, control)
            # print(clip_out)

            samples = self.ddim.run(img, context, hint, scale)

            # x_samples = self.model.decode_first_stage(samples)
            # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            x_samples = self.vae.run(samples).cpu().numpy()
            results = [x_samples[i] for i in range(num_samples)]
        return results