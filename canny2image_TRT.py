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

from cuda import cudart
from transformers import CLIPTokenizer

from trt.clip_trt import CLIPTrt
from trt.vae_trt import VAETrt
from trt.ddim_trt import DDIMTrt

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
import ctypes
ctypes.CDLL('./trt/libmyplugins.so.1', mode=ctypes.RTLD_GLOBAL)

trt.init_libnvinfer_plugins(trt_logger, '')

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        ddim_timesteps = np.asarray(list(range(0, 1000, 50))) + 1
        alphacums = self.model.alphas_cumprod.cpu().numpy()
        self.ddim_alphas = alphacums[ddim_timesteps]
        self.ddim_alphas_sqrt = np.sqrt(self.ddim_alphas)
        self.ddim_alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
        self.ddim_alphas_prev_sqrt = np.sqrt(self.ddim_alphas_prev)
        self.ddim_alphas_prev_sub_sqrt = np.sqrt(1. - self.ddim_alphas_prev)
        self.ddim_sigmas = 0 * self.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)

        # del model

        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        
        self.device = torch.device('cuda')

        self.tensors = {
            'input_ids': torch.ones(size=(2, 77), dtype=torch.int32, device=self.device),
            'control': torch.ones(size=(1, 256, 384, 3), dtype=torch.float32, device=self.device),
            'context': torch.ones(size=(2, 77, 8 * 4560), dtype=torch.float32, device=self.device),
            'hint': torch.ones(size=(1, 320, 32, 48), dtype=torch.float32, device=self.device),

            'z': torch.ones(size=(1, 4, 32, 48), dtype=torch.float, device=self.device),
            'img_out': torch.ones(size=(1, 256, 384, 3), dtype=torch.uint8, device=self.device),

            'x': torch.ones(size=(1, 4, 32, 48), dtype=torch.float32, device=self.device),
            't_emb': torch.ones(size=(1,), dtype=torch.int32, device=self.device),
            'out': torch.ones(size=(2, 4, 32, 48), dtype=torch.float, device=self.device),
            'c1': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'c2': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'c3': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'c4': torch.ones(size=(2, 320, 16, 24), dtype=torch.float, device=self.device),
            'c5': torch.ones(size=(2, 640, 16, 24), dtype=torch.float, device=self.device),
            'c6': torch.ones(size=(2, 640, 16, 24), dtype=torch.float, device=self.device),
            'c7': torch.ones(size=(2, 640, 8, 12), dtype=torch.float, device=self.device),
            'c8': torch.ones(size=(2, 1280, 8, 12), dtype=torch.float, device=self.device),
            'c9': torch.ones(size=(2, 1280, 8, 12), dtype=torch.float, device=self.device),
            'c10': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'c11': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'c12': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'c13': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),

            'h1': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'h2': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'h3': torch.ones(size=(2, 320, 32, 48), dtype=torch.float, device=self.device),
            'h4': torch.ones(size=(2, 320, 16, 24), dtype=torch.float, device=self.device),
            'h5': torch.ones(size=(2, 640, 16, 24), dtype=torch.float, device=self.device),
            'h6': torch.ones(size=(2, 640, 16, 24), dtype=torch.float, device=self.device),
            'h7': torch.ones(size=(2, 640, 8, 12), dtype=torch.float, device=self.device),
            'h8': torch.ones(size=(2, 1280, 8, 12), dtype=torch.float, device=self.device),
            'h9': torch.ones(size=(2, 1280, 8, 12), dtype=torch.float, device=self.device),
            'h10': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'h11': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'h12': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
            'h13': torch.ones(size=(2, 1280, 4, 6), dtype=torch.float, device=self.device),
        }

        self.engine_context_map = {}
        self.stream = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]  
        self.stream1 = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]
        self.event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)[1]
        self.event1 = cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)[1]

        self.clip_instance = self.load_engine('sd_clip', self.stream1)
        # self.hint_instance = self.load_engine('sd_hint', self.stream)
        # self.vae_instance = self.load_engine('sd_vae_fp16_native', self.stream)
        self.vae_instance = self.load_engine('sd_vae_fp16', self.stream)

        # self.vae_instance = self.load_engine('sd_vae', self.stream)

        # self.control_graph_instance = self.load_engine('sd_control', self.stream)
        # self.unet_input_graph_instance = self.load_engine('sd_unet_input', self.stream)
        # self.unet_output_graph_instance = self.load_engine('sd_unet_output', self.stream)

        self.control_fp16_graph_instance = self.load_engine('sd_control_fp16', self.stream1)
        self.unet_input_fp16_graph_instance = self.load_engine('sd_unet_input_fp16', self.stream)
        self.unet_output_fp16_graph_instance = self.load_engine('sd_unet_output_fp16', self.stream)

        torch.cuda.set_stream(torch.cuda.ExternalStream(int(self.stream)))

    def load_engine(self, model, stream):
        trt_logger = trt.Logger(trt.Logger.INFO)

        with open('trt/{}.plan'.format(model), 'rb') as f, trt.Runtime(trt_logger) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
            trt_ctx = trt_engine.create_execution_context()

            for index in range(trt_engine.num_io_tensors):
                name = trt_engine.get_binding_name(index)
                if 'vae' in model:
                    if name == 'z':
                        trt_ctx.set_tensor_address(name, self.tensors['x'].data_ptr())
                    if name == 'out':
                        trt_ctx.set_tensor_address(name, self.tensors['img_out'].data_ptr())
                else:
                    trt_ctx.set_tensor_address(name, self.tensors[name].data_ptr())

            trt_ctx.execute_async_v3(stream)
            
            cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            trt_ctx.execute_async_v3(stream)
            graph = cudart.cudaStreamEndCapture(stream)[1]
            graph_instance = cudart.cudaGraphInstantiate(graph, 0)[1]
            # graph_instance = None
            self.engine_context_map[model] = trt_ctx
        return graph_instance

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.unsqueeze(control, 0)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            img = torch.randn((1, 4, 32, 48), device=control.device)

            text = [prompt + ', ' + a_prompt, n_prompt]
            batch_encoding = self.tokenizer(text, truncation=True, max_length=77, return_attention_mask=False, return_length=False,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            self.tensors["input_ids"].copy_(batch_encoding["input_ids"])
            cudart.cudaEventRecord(self.event, self.stream)
            cudart.cudaStreamWaitEvent(self.stream1, self.event, cudart.cudaEventWaitDefault)
            cudart.cudaGraphLaunch(self.clip_instance, self.stream1)
            cudart.cudaEventRecord(self.event1, self.stream1)
            # cudart.cudaGraphLaunch(self.hint_instance, self.stream)

            import einops
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            hint = self.model.control_model.input_hint_block(control, None, None)
            self.tensors['hint'].copy_(hint)  
            
            cudart.cudaStreamWaitEvent(self.stream, self.event1, cudart.cudaEventWaitDefault)

            for index in reversed(range(20)):
                t_emb = torch.full((1, ), index, device=self.device, dtype=torch.int32)
                self.tensors['t_emb'].copy_(t_emb)
                self.tensors['x'].copy_(img)
                cudart.cudaEventRecord(self.event, self.stream)
                
                cudart.cudaStreamWaitEvent(self.stream1, self.event, cudart.cudaEventWaitDefault)

                # if index == 19 or index % 3 == 0:
                # if index > 10:
                if index > 15 or index % 3 == 0:
                    cudart.cudaGraphLaunch(self.control_fp16_graph_instance, self.stream1)
                cudart.cudaEventRecord(self.event1, self.stream1)

                if index > 12 or index % 2 == 0:
                # if index > 4:
                    cudart.cudaGraphLaunch(self.unet_input_fp16_graph_instance, self.stream)

                cudart.cudaStreamWaitEvent(self.stream, self.event1, cudart.cudaEventWaitDefault)
                cudart.cudaGraphLaunch(self.unet_output_fp16_graph_instance, self.stream)

                model_t, model_uncond = self.tensors['out'].chunk(2)
                model_output = model_uncond + scale * (model_t - model_uncond)

                e_t = model_output

                pred_x0 = (self.tensors['x'] - self.ddim_sqrt_one_minus_alphas[index] * e_t) / self.ddim_alphas_sqrt[index]
                dir_xt = self.ddim_alphas_prev_sub_sqrt[index] * e_t
                img = self.ddim_alphas_prev_sqrt[index] * pred_x0 + dir_xt
            
            cudart.cudaGraphLaunch(self.vae_instance, self.stream)


            cudart.cudaStreamSynchronize(self.stream)
            cudart.cudaStreamSynchronize(self.stream1)
            x_samples = self.tensors['img_out'].cpu().numpy()

            results = [x_samples[i] for i in range(num_samples)]
        return results