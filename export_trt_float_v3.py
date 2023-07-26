import os
import sys

import numpy as np
import tensorrt as trt
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding
from cldm.model import create_model, load_state_dict

from transformers import CLIPTextModel

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()

control_model = model.control_model
unet = model.model.diffusion_model

trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')

def compute_embedding():
    control_model = model.control_model
    unet = model.model.diffusion_model

    embed = np.asarray(list(range(0, 1000, 50))) + 1
    embed = torch.from_numpy(embed).type(torch.long).cuda()
    embed = timestep_embedding(embed, 320, repeat_only=False)

    # control_net
    with torch.no_grad():
        e = control_model.time_embed(embed)

        # input blocks
        index = [1, 2, 4, 5, 7, 8, 10, 11]
        for i in index:
            o = control_model.input_blocks[i][0].emb_layers(e)
            control_model.input_blocks[i][0].emb_layers.register_buffer('table', o)

        # middle blocks
        index = [0, 2]
        for i in index:
            o = control_model.middle_block[i].emb_layers(e)
            control_model.middle_block[i].emb_layers.register_buffer('table', o)
    
    # unet
    with torch.no_grad():
        e = unet.time_embed(embed)

        # input blocks
        index = [1, 2, 4, 5, 7, 8, 10, 11]
        for i in index:
            o = unet.input_blocks[i][0].emb_layers(e)
            unet.input_blocks[i][0].emb_layers.register_buffer('table', o)

        # middle blocks
        index = [0, 2]
        for i in index:
            o = unet.middle_block[i].emb_layers(e)
            unet.middle_block[i].emb_layers.register_buffer('table', o)
        
        # output blocks
        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for i in index:
            o = unet.output_blocks[i][0].emb_layers(e)
            unet.output_blocks[i][0].emb_layers.register_buffer('table', o)

def create_unet_output_engine():
    unet = model.model.diffusion_model
    device = torch.device("cuda")
    x = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to(device)
    t_emb = torch.zeros(1, dtype=torch.int64).to(device)
    context = torch.zeros(2, 77, 8 * 4560, dtype=torch.float32).to(device)

    control = []
    control.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 320, 16, 24, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 640, 16, 24, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 640, 16, 24, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 640, 8, 12, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 8, 12, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 8, 12, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    control.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))

    hs = []
    hs.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 320, 32, 48, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 320, 16, 24, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 640, 16, 24, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 640, 16, 24, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 640, 8, 12, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 8, 12, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 8, 12, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))
    hs.append(torch.zeros(2, 1280, 4, 6, dtype=torch.float32).to(device))

    output_names = ['out']
    input_names = ['x', 't_emb', 'context']
    for i in range(13):
        input_names.append('c{}'.format(i + 1))
    for i in range(13):
        input_names.append('h{}'.format(i + 1))

    torch.onnx.export(unet,
                      (x, t_emb, context, {'control' : control, 'hs' : hs}),
                      'trt/sd_unet_output.onnx',
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=input_names,
                      output_names=output_names)
    
    from export_state import global_state
    print(global_state['start'])

    # convert_onnx_to_trt('trt/sd_unet_output.onnx', 2<<30, 'trt/sd_unet_output_fp16.plan')
    # exit(0)
    # os.system("trtexec --onnx=trt/sd_unet_output.onnx --saveEngine=trt/sd_unet_output.plan")
    os.system("trtexec --onnx=trt/sd_unet_output.onnx --saveEngine=trt/sd_unet_output_fp16.plan --fp16")
    

def create_unet_input_engine():
    unet = model.model.diffusion_model
    device = torch.device("cuda")
    x = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to(device)
    t_emb = torch.zeros(1, dtype=torch.int64).to(device)
    context = torch.zeros(2, 77, 8 * 4560, dtype=torch.float32).to(device)

    output_names = ['h{}'.format(i + 1) for i in range(13)]

    torch.onnx.export(unet,
                      (x, t_emb, context),
                      'trt/sd_unet_input.onnx',
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=['x', 't_emb', 'context'],
                      output_names=output_names)
    
    from export_state import global_state
    print(global_state['start'])
    convert_onnx_to_trt('trt/sd_unet_input.onnx', 2<<30, 'trt/sd_unet_input_fp16.plan')
    # exit(0)
    # os.system("trtexec --onnx=trt/sd_unet_input.onnx --saveEngine=trt/sd_unet_input.plan")
    # os.system("trtexec --onnx=trt/sd_unet_input.onnx --saveEngine=trt/sd_unet_input_fp16.plan --fp16")

def create_control_engine():
    control_model = model.control_model
    device = torch.device("cuda")
    x = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to(device)
    hint = torch.zeros(1, 320, 32, 48, dtype=torch.float32).to(device)
    t_emb = torch.zeros(1, dtype=torch.int64).to(device)
    context = torch.zeros(2, 77, 8 * 4560, dtype=torch.float32).to(device)

    output_names = ['c{}'.format(i + 1) for i in range(13)]

    torch.onnx.export(control_model,
                      (x, hint, t_emb, context),
                      'trt/sd_control.onnx',
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=['x', 'hint', 't_emb', 'context'],
                      output_names=output_names)
    
    from export_state import global_state
    print(global_state['start'])

    convert_onnx_to_trt('trt/sd_control.onnx', 2<<30, 'trt/sd_control_fp16.plan')
    # exit(0)
    # os.system("trtexec --onnx=trt/sd_control.onnx --saveEngine=trt/sd_control.plan")
    # os.system("trtexec --onnx=trt/sd_control.onnx --saveEngine=trt/sd_control_fp16.plan --fp16")

class CLIP(torch.nn.Module):
    def __init__(self, control_model, unet):
        super().__init__()
        self.control_model = control_model
        self.unet = unet
        self.transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").text_model
        mask = torch.empty(2, 77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        self.register_buffer("causal_mask", mask)

    def forward(self, input_ids, control):
        hidden_states = self.transformer.embeddings(input_ids=input_ids)
        encoder_outputs = self.transformer.encoder(
                inputs_embeds=hidden_states,
                causal_attention_mask=self.causal_mask)
        context = self.transformer.final_layer_norm(encoder_outputs[0])

        # nhwc -> nchw
        hint = control.permute(0, 3, 1, 2)
        hint = control_model.input_hint_block(hint, None, None)

        context_bank = []
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
        
        # for i in range(len(context_bank)):
        #     context_bank[i] = context_bank[i].reshape(2, 77, 8, -1).permute(0, 2, 1, 3).reshape(16, 77, -1)
        context_bank = torch.cat(context_bank, -1)

        return context_bank, hint

def create_clip_engine():
    clip = CLIP(control_model, unet).cuda()
    device = torch.device("cuda")
    input_ids = torch.ones(2, 77, dtype=torch.int64).to(device)
    control = torch.ones(1, 256, 384, 3, dtype=torch.float32).to(device)

    torch.onnx.export(clip,
                      (input_ids, control),
                      'trt/sd_clip.onnx',
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=['input_ids', 'control'],
                      output_names=['context', 'hint'])
    
    os.system("trtexec --onnx=trt/sd_clip.onnx --saveEngine=trt/sd_clip.plan --workspace=1000")
    # os.system("trtexec --onnx=trt/sd_clip.onnx --saveEngine=trt/sd_clip_fp16.plan --workspace=1000 --fp16")

class VAE(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, z):
        z = 1. / model.scale_factor * z
        z = self.vae.decode(z)
        z = z.permute(0, 2, 3, 1) * 127.5 + 127.5
        z = z.clip(0, 255).type(torch.uint8)
        return z

def create_vae_engine():
    vae = VAE(model.first_stage_model).cuda()
    device = torch.device("cuda")
    z = torch.ones(1, 4, 32, 48, dtype=torch.float).to(device)

    torch.onnx.export(vae,
                      (z,),
                      'trt/sd_vae.onnx',
                      export_params=True,
                      opset_version=18,
                      do_constant_folding=True,
                      keep_initializers_as_inputs=True,
                      input_names=['z'],
                      output_names=['out'])
    
    # convert_onnx_to_trt('trt/sd_vae.onnx', 2<<30, 'trt/sd_vae_fp16.plan')
    # os.system("trtexec --onnx=trt/sd_vae.onnx --saveEngine=trt/sd_vae.plan --workspace=2000")
    os.system("trtexec --onnx=trt/sd_vae.onnx --saveEngine=trt/sd_vae_fp16.plan --workspace=2000 --fp16")

def convert_onnx_to_trt(onnx, workspace, filename):
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    parser.flags = 1 << (int)(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

    success = parser.parse_from_file(onnx)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if 'fp16' in filename:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(filename, 'wb') as f:
        f.write(serialized_engine)

def convert_trt_engine():
    # vae
    create_vae_engine()

    # clip
    create_clip_engine()
    
    compute_embedding()
    from export_state import global_state

    # control
    global_state['start'] = 0
    create_control_engine()

    global_state['start'] = 1440
    create_unet_input_engine()

    global_state['start'] = 2880
    create_unet_output_engine()

convert_trt_engine()
