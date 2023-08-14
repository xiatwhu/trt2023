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

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
import ctypes
ctypes.CDLL('./trt/libmyplugins.so.1', mode=ctypes.RTLD_GLOBAL)

trt.init_libnvinfer_plugins(trt_logger, '')

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

useGraph = True

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()

        self.device = torch.device('cuda')

        ddim_timesteps = np.asarray(list(range(0, 1000, 50))) + 1
        alphacums = self.model.alphas_cumprod.cpu().numpy()
        self.ddim_alphas = alphacums[ddim_timesteps]
        self.ddim_alphas_sqrt = np.sqrt(self.ddim_alphas)
        self.ddim_alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
        self.ddim_alphas_prev_sqrt = np.sqrt(self.ddim_alphas_prev)
        self.ddim_alphas_prev_sub_sqrt = np.sqrt(1. - self.ddim_alphas_prev)
        self.ddim_sigmas = 0 * self.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)

        self.sqrt_one_minus_at = []
        self.sqrt_one_minus_a_prev = []
        self.sqrt_a_prev = []
        self.sqrt_at = []
        self.t_emb = []

        for index in range(20):
            a_t = torch.full((1, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((1, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            self.sqrt_one_minus_at.append(torch.full((1, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=self.device))
            self.sqrt_one_minus_a_prev.append((1. - a_prev).sqrt())
            self.sqrt_a_prev.append(a_prev.sqrt())
            self.sqrt_at.append(a_t.sqrt())
            self.t_emb.append(torch.full((1, ), index, device=self.device, dtype=torch.int32))

        self.a0 = self.ddim_alphas_prev_sqrt / self.ddim_alphas_sqrt
        self.a1 = self.ddim_alphas_prev_sub_sqrt - self.ddim_alphas_prev_sqrt * self.ddim_sqrt_one_minus_alphas / self.ddim_alphas_sqrt

        self.run_step = [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        # del model

        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

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
        self.vae_instance = self.load_engine('sd_vae_fp16', self.stream)

        self.control_fp16_graph_instance = self.load_engine('sd_control_fp16', self.stream1)
        self.unet_input_fp16_graph_instance = self.load_engine('sd_unet_input_fp16', self.stream)
        self.unet_output_fp16_graph_instance = self.load_engine('sd_unet_output_fp16', self.stream)

        torch.cuda.set_stream(torch.cuda.ExternalStream(int(self.stream)))

        import time
        time.sleep(60 * 2)

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
            
            if useGraph:
                cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                trt_ctx.execute_async_v3(stream)
                graph = cudart.cudaStreamEndCapture(stream)[1]
                graph_instance = cudart.cudaGraphInstantiate(graph, 0)[1]
            else:
                graph_instance = None

            self.engine_context_map[model] = trt_ctx
        return graph_instance

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.unsqueeze(control, 0)
            # print('canny: ', control.sum(), detected_map.sum() / 255.0)

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
            if useGraph:
                cudart.cudaGraphLaunch(self.clip_instance, self.stream1)
            else:
                self.engine_context_map['sd_clip'].execute_async_v3(self.stream1)

            cudart.cudaEventRecord(self.event1, self.stream1)
            # cudart.cudaGraphLaunch(self.hint_instance, self.stream)

            import einops
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            hint = self.model.control_model.input_hint_block(control, None, None)
            self.tensors['hint'].copy_(hint)  
            
            cudart.cudaStreamWaitEvent(self.stream, self.event1, cudart.cudaEventWaitDefault)

            for index in reversed(range(20)):
                self.tensors['x'].copy_(img)

                # if True:
                ## 跑 12 个 step
                # if index in [19, 18, 17, 16, 15, 14, 13, 12, 9, 6, 3, 0]: # 4.993117427548901
                # if index in [19, 18, 17, 16, 15, 14, 13, 11, 9, 6, 3, 0]: # 4.823739027404583
                # if index in [19, 18, 17, 16, 15, 14, 13, 12, 11, 9, 6, 3]:    # 4.596818284133821
                # if index in [19, 18, 17, 16, 15, 14, 13, 12, 10, 8, 6, 3]:      # 4.503659897417926

                ## 跑 11 个 step
                # if index in [19, 18, 17, 16, 15, 14, 12, 9, 6, 3, 0]: # 5.3313205394989325
                # if index in [19, 18, 17, 16, 15, 14, 13, 11, 8, 4, 0]: # 5.249650220918213
                # if index in [19, 18, 17, 16, 15, 14, 13, 11, 9, 6, 3]: # 4.824511678920353
                # if index in [19, 18, 17, 16, 15, 14, 13, 12, 9, 6, 3]: # 4.824511678920353
                # if index in [19, 18, 17, 16, 15, 14, 13, 12, 10, 7, 4]: # 4.824511678920353
                # if index in [19, 18, 17, 16, 15, 14, 13, 11, 9, 7, 3]:  # 5.00344845752016

                ## 跑 10 个 step
                if index in [19, 18, 17, 16, 15, 14, 13, 11, 8, 4]: # 5.287159632408505
                # if index in [19, 18, 17, 16, 15, 14, 13, 10, 7, 3]:   # 5.426477961956595
                # if index in [19, 18, 17, 16, 15, 14, 12, 9, 6, 3]:   # 5.493245796290085

                ## 跑 9 个 step
                # if index in [19, 18, 17, 16, 15, 14, 12, 10, 5]: # 6.021245215522073
                # if index in [19, 18, 17, 16, 15, 14, 13, 10, 5]: # 5.811215823737912
                # if index in [19, 18, 17, 16, 15, 14, 12, 8, 4]: # 5.610201291122897
                # if index in [19, 18, 17, 16, 15, 14, 12, 9, 5]: # 5.724937136374654
                # if index in [19, 18, 17, 16, 15, 13, 11, 8, 4]:
                # if index in self.run_step:
                    
                    self.tensors['t_emb'].copy_(self.t_emb[index])
                    cudart.cudaEventRecord(self.event, self.stream)
                    cudart.cudaStreamWaitEvent(self.stream1, self.event, cudart.cudaEventWaitDefault)

                    if useGraph:
                        cudart.cudaGraphLaunch(self.control_fp16_graph_instance, self.stream1)
                        cudart.cudaEventRecord(self.event1, self.stream1)
                        cudart.cudaGraphLaunch(self.unet_input_fp16_graph_instance, self.stream)
                        cudart.cudaStreamWaitEvent(self.stream, self.event1, cudart.cudaEventWaitDefault)
                        cudart.cudaGraphLaunch(self.unet_output_fp16_graph_instance, self.stream)
                    else:
                        self.engine_context_map['sd_control_fp16'].execute_async_v3(self.stream1)
                        cudart.cudaEventRecord(self.event1, self.stream1)
                        self.engine_context_map['sd_unet_input_fp16'].execute_async_v3(self.stream)
                        cudart.cudaStreamWaitEvent(self.stream, self.event1, cudart.cudaEventWaitDefault)
                        self.engine_context_map['sd_unet_output_fp16'].execute_async_v3(self.stream)

                    model_t, model_uncond = self.tensors['out'].chunk(2)
                    model_output = model_uncond + scale * (model_t - model_uncond)

                e_t = model_output
                # pred_x0 = (self.tensors['x'] - self.ddim_sqrt_one_minus_alphas[index] * e_t) / self.ddim_alphas_sqrt[index]
                # dir_xt = self.ddim_alphas_prev_sub_sqrt[index] * e_t
                # img = self.ddim_alphas_prev_sqrt[index] * pred_x0 + dir_xt
                # img = self.a0[index] * self.tensors['x'] + self.a1[index] * model_output
            
                # a_t = torch.full((1, 1, 1, 1), self.ddim_alphas[index], device=self.device)
                # a_prev = torch.full((1, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
                # sqrt_one_minus_at = torch.full((1, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)
                # sqrt_one_minus_a_prev = (1. - a_prev).sqrt()
                # sqrt_a_prev = a_prev.sqrt()
                # sqrt_at = a_t.sqrt()

                pred_x0 = (self.tensors['x'] - self.sqrt_one_minus_at[index] * e_t) / self.sqrt_at[index]
                dir_xt = self.sqrt_one_minus_a_prev[index] * e_t
                img = self.sqrt_a_prev[index] * pred_x0 + dir_xt

            self.tensors['x'].copy_(img)
            if useGraph:
                cudart.cudaGraphLaunch(self.vae_instance, self.stream)
            else:
                self.engine_context_map['sd_vae_fp16'].execute_async_v3(self.stream)

            cudart.cudaStreamSynchronize(self.stream)
            cudart.cudaStreamSynchronize(self.stream1)
            x_samples = self.tensors['img_out'].cpu().numpy()

            results = [x_samples[i] for i in range(num_samples)]
        return results