import tensorrt as trt
from cuda import cudart
import torch
import numpy as np

trt_logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(trt_logger, '')

class DDIMTrt(object):
    def __init__(self, model):
        super().__init__()

        self.model = model
        ddim_timesteps = np.asarray(list(range(0, 1000, 50))) + 1
        alphacums = self.model.alphas_cumprod.cpu().numpy()

        self.ddim_alphas = alphacums[ddim_timesteps]
        self.ddim_alphas_sqrt = np.sqrt(self.ddim_alphas)
        self.ddim_alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
        self.ddim_alphas_prev_sqrt = np.sqrt(self.ddim_alphas_prev)
        self.ddim_alphas_prev_sub_sqrt = np.sqrt(1. - self.ddim_alphas_prev)
        self.ddim_sigmas = 0 * self.ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)

        device = torch.device('cuda')
        tensors_shape_map = {
                "x": (1, 4, 32, 48),
                "hint": (1, 320, 32, 48),
                "t_emb": (1, ),
                "context": (16, 77, 4560),
                "out": (2, 4, 32, 48),
                "c1": (2, 320, 32, 48),
                "c2": (2, 320, 32, 48),
                "c3": (2, 320, 32, 48),
                "c4": (2, 320, 16, 24),
                "c5": (2, 640, 16, 24),
                "c6": (2, 640, 16, 24),
                "c7": (2, 640, 8, 12),
                "c8": (2, 1280, 8, 12),
                "c9": (2, 1280, 8, 12),
                "c10": (2, 1280, 4, 6),
                "c11": (2, 1280, 4, 6),
                "c12": (2, 1280, 4, 6),
                "c13": (2, 1280, 4, 6),
                "h1": (2, 320, 32, 48),
                "h2": (2, 320, 32, 48),
                "h3": (2, 320, 32, 48),
                "h4": (2, 320, 16, 24),
                "h5": (2, 640, 16, 24),
                "h6": (2, 640, 16, 24),
                "h7": (2, 640, 8, 12),
                "h8": (2, 1280, 8, 12),
                "h9": (2, 1280, 8, 12),
                "h10": (2, 1280, 4, 6),
                "h11": (2, 1280, 4, 6),
                "h12": (2, 1280, 4, 6),
                "h13": (2, 1280, 4, 6),
        }

        tensors = {}
        for k in tensors_shape_map.keys():
            dtype = torch.int32 if k == 't_emb' else torch.float32
            tensors[k] = torch.ones(size=tensors_shape_map[k], dtype=dtype, device=device).contiguous()

        torch.cuda.synchronize()

        self.tensors = tensors
        self.stream0 = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]
        self.stream1 = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]
        self.engine_context_map = {}

        self.control_graph_instance = self.load_engine('sd_control', self.stream0)
        self.unet_input_graph_instance = self.load_engine('sd_unet_input', self.stream1)
        self.unet_output_graph_instance = self.load_engine('sd_unet_output', self.stream1)

        self.control_fp16_graph_instance = self.load_engine('sd_control_fp16', self.stream0)
        self.unet_input_fp16_graph_instance = self.load_engine('sd_unet_input_fp16', self.stream1)
        self.unet_output_fp16_graph_instance = self.load_engine('sd_unet_output_fp16', self.stream1)
        

    def load_engine(self, model, stream):
        trt_logger = trt.Logger(trt.Logger.INFO)

        with open('trt/{}.plan'.format(model), 'rb') as f, trt.Runtime(trt_logger) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
            trt_ctx = trt_engine.create_execution_context()

            for index in range(trt_engine.num_io_tensors):
                name = trt_engine.get_binding_name(index)
                trt_ctx.set_tensor_address(name, self.tensors[name].data_ptr())

            trt_ctx.execute_async_v3(stream)
            
            cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            trt_ctx.execute_async_v3(stream)
            graph = cudart.cudaStreamEndCapture(stream)[1]
            graph_instance = cudart.cudaGraphInstantiate(graph, 0)[1]
            # graph_instance = None
            self.engine_context_map[model] = trt_ctx
        return graph_instance

    def run(self, x, context, hint, unconditional_guidance_scale):
        device = x.device
        with torch.no_grad():
            for index in reversed(range(20)):
            
                if index == 19:
                    self.tensors['hint'].copy_(hint)
                    self.tensors['context'].copy_(context)

                t_emb = torch.full((1,), index, device=device, dtype=torch.int32)
                self.tensors['x'].copy_(x)
                self.tensors['t_emb'].copy_(t_emb)
                torch.cuda.synchronize()

                if index < 12:
                    cudart.cudaGraphLaunch(self.control_fp16_graph_instance, self.stream0)
                    cudart.cudaGraphLaunch(self.unet_input_fp16_graph_instance, self.stream1)
                    cudart.cudaStreamSynchronize(self.stream0)
                    # cudart.cudaStreamSynchronize(self.stream1)
                    cudart.cudaGraphLaunch(self.unet_output_fp16_graph_instance, self.stream1)
                    cudart.cudaStreamSynchronize(self.stream1)
                else:
                    cudart.cudaGraphLaunch(self.control_graph_instance, self.stream0)
                    cudart.cudaGraphLaunch(self.unet_input_graph_instance, self.stream1)
                    cudart.cudaStreamSynchronize(self.stream0)
                    # cudart.cudaStreamSynchronize(self.stream1)
                    cudart.cudaGraphLaunch(self.unet_output_graph_instance, self.stream1)
                    cudart.cudaStreamSynchronize(self.stream1)
                model_t, model_uncond = self.tensors['out'].chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

                e_t = model_output

                pred_x0 = (x - self.ddim_sqrt_one_minus_alphas[index] * e_t) / self.ddim_alphas_sqrt[index]
                dir_xt = self.ddim_alphas_prev_sub_sqrt[index] * e_t
                x = self.ddim_alphas_prev_sqrt[index] * pred_x0 + dir_xt

        return x
