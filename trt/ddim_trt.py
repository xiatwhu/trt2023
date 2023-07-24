import tensorrt as trt
from cuda import cudart
import torch

class DDIMTrt(object):
    def __init__(self, model):
        super().__init__()

        self.model = model

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

        self.control_graph_instance = self.load_engine('control', self.stream0)
        self.unet_input_graph_instance = self.load_engine('unet_input', self.stream1)
        self.unet_output_graph_instance = self.load_engine('unet_output', self.stream1)

        self.control_fp16_graph_instance = self.load_engine('control_fp16', self.stream0)
        self.unet_input_fp16_graph_instance = self.load_engine('unet_input_fp16', self.stream1)
        self.unet_output_fp16_graph_instance = self.load_engine('unet_output_fp16', self.stream1)
        

    def load_engine(self, model, stream):
        trt_logger = trt.Logger(trt.Logger.VERBOSE)

        with open('trt/{}.plan'.format(model), 'rb') as f, trt.Runtime(trt_logger) as runtime:
            trt_engine = runtime.deserialize_cuda_engine(f.read())
            trt_ctx = trt_engine.create_execution_context()

            for index in range(trt_engine.num_io_tensors):
                name = trt_engine.get_binding_name(index)
                trt_ctx.set_tensor_address(name, self.tensors[name].data_ptr())
            # print(trt_engine.num_bindings)
            # print(trt_ctx.all_binding_shapes_specified)
            # print(trt_ctx.all_shape_inputs_specified)
            # print(trt_ctx.infer_shapes())
            # # exit(0)
            # print(cudart.cudaStreamSynchronize(stream))
            trt_ctx.execute_async_v3(stream)
            
            cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
            trt_ctx.execute_async_v3(stream)
            graph = cudart.cudaStreamEndCapture(stream)[1]
            graph_instance = cudart.cudaGraphInstantiate(graph, 0)[1]
            # graph_instance = None
            self.engine_context_map[model] = trt_ctx
        return graph_instance

    def run(self, x, c, index, unconditional_guidance_scale, unconditional_conditioning):
        device = x.device
        with torch.no_grad():
            
            if index == 19:
                control = torch.cat(c['c_concat'], 1)
                hint = self.model.control_model.input_hint_block(control, None, None)
                self.tensors['hint'].copy_(hint)

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
                context_bank = torch.cat(context_bank, -1)

                self.tensors['context'].copy_(context_bank)

            # t_emb = timestep_embedding(t, 320, repeat_only=False)
            t_emb = torch.full((1,), index, device=device, dtype=torch.int32)
            self.tensors['x'].copy_(x)
            self.tensors['t_emb'].copy_(t_emb)
            torch.cuda.synchronize()

            if index < 12:
                # self.engine_context_map['unet_input_fp16'].execute_async_v3(self.stream1)
                # self.engine_context_map['control_fp16'].execute_async_v3(self.stream0)
                # cudart.cudaStreamSynchronize(self.stream0)
                # cudart.cudaStreamSynchronize(self.stream1)
                # self.engine_context_map['unet_output_fp16'].execute_async_v3(self.stream1)
                # cudart.cudaStreamSynchronize(self.stream1)
                cudart.cudaGraphLaunch(self.control_fp16_graph_instance, self.stream0)
                cudart.cudaGraphLaunch(self.unet_input_fp16_graph_instance, self.stream1)
                cudart.cudaStreamSynchronize(self.stream0)
                cudart.cudaStreamSynchronize(self.stream1)
                cudart.cudaGraphLaunch(self.unet_output_fp16_graph_instance, self.stream1)
                cudart.cudaStreamSynchronize(self.stream1)
            else:
                self.engine_context_map['unet_input'].execute_async_v3(self.stream1)
                self.engine_context_map['control'].execute_async_v3(self.stream0)
                cudart.cudaStreamSynchronize(self.stream0)
                cudart.cudaStreamSynchronize(self.stream1)
                self.engine_context_map['unet_output'].execute_async_v3(self.stream1)
                cudart.cudaStreamSynchronize(self.stream1)
                # cudart.cudaGraphLaunch(self.control_graph_instance, self.stream0)
                # cudart.cudaGraphLaunch(self.unet_input_graph_instance, self.stream1)
                # cudart.cudaStreamSynchronize(self.stream0)
                # cudart.cudaStreamSynchronize(self.stream1)
                # cudart.cudaGraphLaunch(self.unet_output_graph_instance, self.stream1)
                # cudart.cudaStreamSynchronize(self.stream1)
            model_t, model_uncond = self.tensors['out'].chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
            return model_output
