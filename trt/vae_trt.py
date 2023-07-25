import tensorrt as trt
from cuda import cudart
import torch

trt_logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(trt_logger, '')

class VAETrt(object):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda')
        self.tensors = {
            'z': torch.ones(size=(1, 4, 32, 48), dtype=torch.float, device=self.device),
            'out': torch.ones(size=(1, 256, 384, 3), dtype=torch.uint8, device=self.device)
        }

        self.engine_context_map = {}
        self.stream = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]  
        self.vae_instance = self.load_engine('sd_vae_fp16', self.stream)

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

    def run(self, z):
        self.tensors["z"].copy_(z)

        # print(self.tensors["input_ids"])
        torch.cuda.synchronize()
        cudart.cudaGraphLaunch(self.vae_instance, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        return self.tensors["out"]