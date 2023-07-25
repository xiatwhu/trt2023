import tensorrt as trt
from cuda import cudart
import torch
from transformers import CLIPTokenizer

trt_logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(trt_logger, '')

class CLIPTrt(object):
    def __init__(self):
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

        self.device = torch.device('cuda')
        self.tensors = {
            'input_ids': torch.ones(size=(2, 77), dtype=torch.int32, device=self.device),
            'control': torch.ones(size=(1, 256, 384, 3), dtype=torch.float32, device=self.device),
            'context': torch.ones(size=(16, 77, 4560), dtype=torch.float32, device=self.device),
            'hint': torch.ones(size=(1, 320, 32, 48), dtype=torch.float32, device=self.device)
        }

        self.engine_context_map = {}
        self.stream = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, 0)[1]  
        self.clip_instance = self.load_engine('sd_clip', self.stream)

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

    def run(self, prompt, a_prompt, n_prompt, control):
        text = [prompt + ', ' + a_prompt, n_prompt]
        batch_encoding = self.tokenizer(text, truncation=True, max_length=77, return_attention_mask=False, return_length=False,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        self.tensors["input_ids"].copy_(batch_encoding["input_ids"])
        self.tensors["control"].copy_(control)
        # self.tensors["attention_mask"].copy_(batch_encoding["attention_mask"])

        # print(self.tensors["input_ids"])
        torch.cuda.synchronize()
        cudart.cudaGraphLaunch(self.clip_instance, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        return self.tensors["context"], self.tensors["hint"]