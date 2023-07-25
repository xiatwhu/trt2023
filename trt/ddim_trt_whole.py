import tensorrt as trt
from cuda import cudart
import torch

class DDIMTrt(object):
    def __init__(self, model):
        super().__init__()

        self.model = model

        trt_logger = trt.Logger(trt.Logger.VERBOSE)
        with open('trt/df.plan', 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.model_trt = runtime.deserialize_cuda_engine(f.read())
            self.model_trt_ctx = self.model_trt.create_execution_context()

        self.context_bank = None

        device = torch.device('cuda')
        tensors = []
        tensors.append(torch.zeros(size=(1, 4, 32, 48), dtype=torch.float32, device=device))
        tensors.append(torch.zeros(size=(1, 320, 32, 48), dtype=torch.float32, device=device))
        tensors.append(torch.ones(size=(1,), dtype=torch.int32, device=device))
        tensors.append(torch.zeros(size=(16, 77, 4560), dtype=torch.float32, device=device))
        tensors.append(torch.zeros(size=(2, 4, 32, 48), dtype=torch.float32, device=device))

        self.tensors = tensors
        self.stream = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]
        self.model_trt_ctx.set_tensor_address('x', tensors[0].data_ptr())
        self.model_trt_ctx.set_tensor_address('hint', tensors[1].data_ptr())
        self.model_trt_ctx.set_tensor_address('t_emb', tensors[2].data_ptr())
        self.model_trt_ctx.set_tensor_address('context', tensors[3].data_ptr())
        self.model_trt_ctx.set_tensor_address('out', tensors[4].data_ptr())

        self.model_trt_ctx.execute_async_v3(self.stream)
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        self.model_trt_ctx.execute_async_v3(self.stream)
        self.graph = cudart.cudaStreamEndCapture(self.stream)[1]
        self.cuda_graph_instance = cudart.cudaGraphInstantiate(self.graph, 0)[1]
     
        with open('trt/df_float.plan', 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.model_trt_f = runtime.deserialize_cuda_engine(f.read())
            self.model_trt_f_ctx = self.model_trt_f.create_execution_context()

        self.model_trt_f_ctx.set_tensor_address('x', tensors[0].data_ptr())
        self.model_trt_f_ctx.set_tensor_address('hint', tensors[1].data_ptr())
        self.model_trt_f_ctx.set_tensor_address('t_emb', tensors[2].data_ptr())
        self.model_trt_f_ctx.set_tensor_address('context', tensors[3].data_ptr())
        self.model_trt_f_ctx.set_tensor_address('out', tensors[4].data_ptr())

        self.model_trt_f_ctx.execute_async_v3(self.stream)
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        self.model_trt_f_ctx.execute_async_v3(self.stream)
        self.graph_f = cudart.cudaStreamEndCapture(self.stream)[1]
        self.cuda_graph_f_instance = cudart.cudaGraphInstantiate(self.graph_f, 0)[1]

    def run(self, x, c, index, unconditional_guidance_scale, unconditional_conditioning):
        device = x.device
        with torch.no_grad():
            
            if index == 19:
                control = torch.cat(c['c_concat'], 1)
                hint = self.model.control_model.input_hint_block(control, None, None)
                self.tensors[1].copy_(hint)

                context = torch.cat([torch.cat(c['c_crossattn'], 1), torch.cat(unconditional_conditioning['c_crossattn'], 1)], 0) 
                context_bank = []
                control_model = self.model.control_model
                unet = self.model.model.diffusion_model

                # control net
                block = [1, 2, 4, 5, 7, 8]
                for i in block:
                    context_bank.append(control_model.input_blocks[i][1].transformer_blocks[0].attn2.to_k(context))
                    context_bank.append(control_model.input_blocks[i][1].transformer_blocks[0].attn2.to_v(context))
                
                context_bank.append(control_model.middle_block[1].transformer_blocks[0].attn2.to_k(context))
                context_bank.append(control_model.middle_block[1].transformer_blocks[0].attn2.to_v(context))

                # unet
                block = [1, 2, 4, 5, 7, 8]
                for i in block:
                    context_bank.append(unet.input_blocks[i][1].transformer_blocks[0].attn2.to_k(context))
                    context_bank.append(unet.input_blocks[i][1].transformer_blocks[0].attn2.to_v(context))

                context_bank.append(unet.middle_block[1].transformer_blocks[0].attn2.to_k(context))
                context_bank.append(unet.middle_block[1].transformer_blocks[0].attn2.to_v(context))

                block = [3, 4, 5, 6, 7, 8, 9, 10, 11]
                for i in block:
                    context_bank.append(unet.output_blocks[i][1].transformer_blocks[0].attn2.to_k(context))
                    context_bank.append(unet.output_blocks[i][1].transformer_blocks[0].attn2.to_v(context))
                
                for i in range(len(context_bank)):
                    context_bank[i] = context_bank[i].reshape(2, 77, 8, -1).permute(0, 2, 1, 3).reshape(16, 77, -1)
                self.context_bank = torch.cat(context_bank, -1)

                self.tensors[3].copy_(self.context_bank)

            # t_emb = timestep_embedding(t, 320, repeat_only=False)
            t_emb = torch.full((1,), index, device=device, dtype=torch.int32)
            self.tensors[0].copy_(x)
            self.tensors[2].copy_(t_emb)
            torch.cuda.synchronize()

            if index < 12:
                cudart.cudaGraphLaunch(self.cuda_graph_instance, self.stream)
                cudart.cudaStreamSynchronize(self.stream)
            else:
                cudart.cudaGraphLaunch(self.cuda_graph_f_instance, self.stream)
                cudart.cudaStreamSynchronize(self.stream)
            model_t, model_uncond = self.tensors[4].chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
            return model_output
