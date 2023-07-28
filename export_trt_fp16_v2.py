import os
import sys

import numpy as np
import tensorrt as trt
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding

import ctypes
ctypes.CDLL('./trt/libmyplugins.so.1', mode=ctypes.RTLD_GLOBAL)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
gn_plugin_creator = trt.get_plugin_registry().get_plugin_creator('GroupNorm', "1")
# PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

EPS = 1e-5

def compute_embedding():
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()

    control_model = model.control_model
    unet = model.model.diffusion_model

    embed = np.asarray(list(range(0, 1000, 50))) + 1
    embed = torch.from_numpy(embed).type(torch.long).cuda()
    embed = timestep_embedding(embed, 320, repeat_only=False)

    # control_net
    embed_weight = []
    with torch.no_grad():
        e = control_model.time_embed(embed)

        # input blocks
        index = [1, 2, 4, 5, 7, 8, 10, 11]
        for i in index:
            o = control_model.input_blocks[i][0].emb_layers(e)
            embed_weight.append(o.detach().cpu().numpy())

        # middle blocks
        index = [0, 2]
        for i in index:
            o = control_model.middle_block[i].emb_layers(e)
            embed_weight.append(o.detach().cpu().numpy())
    
    # unet
    with torch.no_grad():
        e = unet.time_embed(embed)

        # input blocks
        index = [1, 2, 4, 5, 7, 8, 10, 11]
        for i in index:
            o = unet.input_blocks[i][0].emb_layers(e)
            embed_weight.append(o.detach().cpu().numpy())

        # middle blocks
        index = [0, 2]
        for i in index:
            o = unet.middle_block[i].emb_layers(e)
            embed_weight.append(o.detach().cpu().numpy())
        
        # output blocks
        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for i in index:
            o = unet.output_blocks[i][0].emb_layers(e)
            embed_weight.append(o.detach().cpu().numpy())

    return embed_weight

def load_torch_model(model_path):
    weight_map = {}

    model = torch.load(model_path, map_location='cpu')

    for k, v in model.items():
        weight_map[k] = v.float().numpy()

    return weight_map

def silu(network, x):
    y = network.add_activation(x.get_output(0), trt.ActivationType.SIGMOID)
    assert y
    x = network.add_elementwise(x.get_output(0), y.get_output(0), trt.ElementWiseOperation.PROD)
    return x

def conv(network, weight_map, x, ch, pre, kernel, padding, stride):
    x = network.add_convolution(
            input=x if isinstance(x, trt.ITensor) else x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(kernel, kernel),
            kernel=weight_map['{}.weight'.format(pre)],
            bias=weight_map['{}.bias'.format(pre)])
    assert x
    x.padding = (padding, padding)
    x.stride = (stride, stride)
    return x

def input_first(network, weight_map, pre, h):
    h = conv(network, weight_map, h, 320, '{}.input_blocks.0.0'.format(pre), 3, 1, 1)
    # h.precision = trt.DataType.FLOAT
    return h

def group_norm(network, weight_map, h, pre, epsilon=EPS, silu=False):
    ch = h.get_output(0).shape[1]
    print("00000000 ", ch)
    # plugin_creator = trt.get_plugin_registry().get_plugin_creator('GroupNorm', "1")
    plugin_creator = gn_plugin_creator
    print("11111111 ", plugin_creator)
    s = network.add_constant([1, ch, 1, 1], weight_map['{}.weight'.format(pre)])
    b = network.add_constant([1, ch, 1, 1], weight_map['{}.bias'.format(pre)])

    print("22222222 ", plugin_creator)
    eps_attr = trt.PluginField("epsilon", np.array([epsilon]), type=trt.PluginFieldType.FLOAT32)
    silu_attr = trt.PluginField("bSwish", np.array([1 if silu else 0]), type=trt.PluginFieldType.INT32)
    field_collection = trt.PluginFieldCollection([eps_attr, silu_attr])

    print("33333333 ", plugin_creator)
    plugin = plugin_creator.create_plugin(name='{}.group_norm'.format(pre), field_collection=field_collection)
    print("44444444 ", plugin_creator, plugin)
    print('group_norm: ', plugin, pre, epsilon, silu)
    n = network.add_plugin_v2(inputs=[h.get_output(0), s.get_output(0), b.get_output(0)], plugin=plugin)

    return n

def layer_norm(network, weight_map, h, pre, epsilon=EPS):
    scale_np = weight_map['{}.weight'.format(pre)]
    ch = scale_np.shape[0]
    scale = network.add_constant([1, 1, ch], scale_np)
    bias_np = weight_map['{}.bias'.format(pre)]
    bias = network.add_constant([1, 1, ch], bias_np)
    n = network.add_normalization(
        h.get_output(0),
        scale=scale.get_output(0),
        bias=bias.get_output(0),
        axesMask=1 << 2)
    assert n
    n.epsilon = epsilon

    return n    

def resblock(network, weight_map, embed_weight, i, ch, h, emb):
    print('resblock: ', h.get_output(0).shape, '{}.in_layers.0'.format(i))
    ## in_layers
    # group_norm
    n = group_norm(network, weight_map, h, '{}.in_layers.0'.format(i), silu=True)
    # silu
    # n = silu(network, n)
    # conv_nd
    n = conv(network, weight_map, n, ch, '{}.in_layers.2'.format(i), 3, 1, 1)

    print('in_layers: ', n.get_output(0).shape)

    ## emb_layers
    m = network.add_constant([20, ch, 1, 1], embed_weight.pop(0))
    m = network.add_gather(m.get_output(0), emb, axis=0)
    print('emb_layers: ', m.get_output(0).shape)

    n = network.add_elementwise(n.get_output(0), m.get_output(0), trt.ElementWiseOperation.SUM)

    ## out_layers
    n = group_norm(network, weight_map, n, '{}.out_layers.0'.format(i), silu=True)
    # n = silu(network, n)
    n = conv(network, weight_map, n, ch, '{}.out_layers.3'.format(i), 3, 1, 1)

    print('out_layers: ', n.get_output(0).shape)

    in_ch = h.get_output(0).shape[1]
    if in_ch != ch:
        # skip_connection
        h = conv(network, weight_map, h, ch, '{}.skip_connection'.format(i), 1, 0, 1)

    h = network.add_elementwise(n.get_output(0), h.get_output(0), trt.ElementWiseOperation.SUM)
    return h

def self_attention(network, weight_map, i, ch, x):   
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5

    wq = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn1.to_q.weight'.format(i)])
    wk = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn1.to_k.weight'.format(i)])
    wv = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn1.to_v.weight'.format(i)])

    q = network.add_matrix_multiply(x.get_output(0), trt.MatrixOperation.NONE,
                                    wq.get_output(0), trt.MatrixOperation.TRANSPOSE)
    k = network.add_matrix_multiply(x.get_output(0), trt.MatrixOperation.NONE,
                                    wk.get_output(0), trt.MatrixOperation.TRANSPOSE)
    v = network.add_matrix_multiply(x.get_output(0), trt.MatrixOperation.NONE,
                                    wv.get_output(0), trt.MatrixOperation.TRANSPOSE)

    # q [2, h * w, c] -> [2, h * w, 8, d] -> [2, 8, h * w, d] -> [16, h * w, d]
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (2, -1, 8, ch // 8)
    q.second_transpose = trt.Permutation([0, 2, 1, 3])
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (16, -1, ch // 8)

    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (2, -1, 8, ch // 8)
    k.second_transpose = trt.Permutation([0, 2, 1, 3])
    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (16, -1, ch // 8)

    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (2, -1, 8, ch // 8)
    v.second_transpose = trt.Permutation([0, 2, 1, 3])
    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (16, -1, ch // 8)

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'b i d, b j d -> b i j')
    print(s.get_output(0).shape)

    s = network.add_scale(s.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([scale], np.float32)))

    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'b i j, b j d -> b i d')
    # [16, h * w, d] -> [2, 8, h * w, d] -> [2, h * w, 8, d] -> [2, h * w, c]
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (2, 8, -1, ch // 8)
    out.second_transpose = trt.Permutation([0, 2, 1, 3])

    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (2, -1, ch)

    # to_out
    outw = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn1.to_out.0.weight'.format(i)])
    outb = network.add_constant((1, 1, ch), weight_map['{}.transformer_blocks.0.attn1.to_out.0.bias'.format(i)])

    out = network.add_matrix_multiply(out.get_output(0), trt.MatrixOperation.NONE,
                                      outw.get_output(0), trt.MatrixOperation.TRANSPOSE)

    out = network.add_elementwise(out.get_output(0), outb.get_output(0), trt.ElementWiseOperation.SUM)

    return out

def cross_attention(network, weight_map, i, ch, x, context):
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5

    wq = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn2.to_q.weight'.format(i)])

    q = network.add_matrix_multiply(x.get_output(0), trt.MatrixOperation.NONE,
                                    wq.get_output(0), trt.MatrixOperation.TRANSPOSE)
    # [2, h*w, c]

    dim = ch // 8
    k = network.add_slice(context['context'],
                          trt.Dims([0, 0, 8 * context['start']]),
                          trt.Dims([2, 77, ch]),
                          trt.Dims([1, 1, 1]))
    v = network.add_slice(context['context'],
                          trt.Dims([0, 0, 8 * (context['start'] + dim)]),
                          trt.Dims([2, 77, ch]),
                          trt.Dims([1, 1, 1]))
    context['start'] += 2 * dim

    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (2, -1, 8, ch // 8)
    q.second_transpose = trt.Permutation([0, 2, 1, 3])
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (16, -1, ch // 8)

    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (2, -1, 8, ch // 8)
    k.second_transpose = trt.Permutation([0, 2, 1, 3])
    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (16, -1, ch // 8)

    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (2, -1, 8, ch // 8)
    v.second_transpose = trt.Permutation([0, 2, 1, 3])
    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (16, -1, ch // 8)

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'b i d, b j d -> b i j')
    print(s.get_output(0).shape)

    # scale = network.add_constant((1, 1, 1), np.array([scale], np.float32))
    # s = network.add_elementwise(s.get_output(0), scale.get_output(0), trt.ElementWiseOperation.PROD)
    s = network.add_scale(s.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([scale], np.float32)))

    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'b i j, b j d -> b i d')
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (2, 8, -1, ch // 8)
    out.second_transpose = trt.Permutation([0, 2, 1, 3])

    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (2, -1, ch)

    # to_out
    outw = network.add_constant((1, ch, ch), weight_map['{}.transformer_blocks.0.attn2.to_out.0.weight'.format(i)])
    outb = network.add_constant((1, 1, ch), weight_map['{}.transformer_blocks.0.attn2.to_out.0.bias'.format(i)])

    out = network.add_matrix_multiply(out.get_output(0), trt.MatrixOperation.NONE,
                                      outw.get_output(0), trt.MatrixOperation.TRANSPOSE)

    out = network.add_elementwise(out.get_output(0), outb.get_output(0), trt.ElementWiseOperation.SUM)

    return out

def feed_forward(network, weight_map, i, ch, x):
    w1 = network.add_constant((1, ch * 8, ch), weight_map['{}.transformer_blocks.0.ff.net.0.proj.weight'.format(i)])
    b1 = network.add_constant((1, 1, ch * 8), weight_map['{}.transformer_blocks.0.ff.net.0.proj.bias'.format(i)])
    n = network.add_matrix_multiply(x.get_output(0), trt.MatrixOperation.NONE,
                                    w1.get_output(0), trt.MatrixOperation.TRANSPOSE)
    n = network.add_elementwise(n.get_output(0), b1.get_output(0), trt.ElementWiseOperation.SUM)

    hw = n.get_output(0).shape[1]
    # w = n.get_output(0).shape[3]
    n1 = network.add_slice(n.get_output(0), trt.Dims([0, 0, 0]), trt.Dims([2, hw, ch * 4]), trt.Dims([1, 1, 1]))
    n2 = network.add_slice(n.get_output(0), trt.Dims([0, 0, ch * 4]), trt.Dims([2, hw, ch * 4]), trt.Dims([1, 1, 1]))

    # gelu
    e = network.add_scale(n2.get_output(0), mode=trt.ScaleMode.UNIFORM, scale=trt.Weights(np.array([2 ** -0.5], np.float32)))
    e = network.add_unary(e.get_output(0), trt.UnaryOperation.ERF)
    e = network.add_scale(e.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([0.5], np.float32)),
                          shift=trt.Weights(np.array([0.5], np.float32)))

    n = network.add_elementwise(n2.get_output(0), e.get_output(0), trt.ElementWiseOperation.PROD)
    n = network.add_elementwise(n.get_output(0), n1.get_output(0), trt.ElementWiseOperation.PROD)

    w2 = network.add_constant((1, ch, ch * 4), weight_map['{}.transformer_blocks.0.ff.net.2.weight'.format(i)])
    b2 = network.add_constant((1, 1, ch), weight_map['{}.transformer_blocks.0.ff.net.2.bias'.format(i)])
    n = network.add_matrix_multiply(n.get_output(0), trt.MatrixOperation.NONE,
                                    w2.get_output(0), trt.MatrixOperation.TRANSPOSE)
    n = network.add_elementwise(n.get_output(0), b2.get_output(0), trt.ElementWiseOperation.SUM)

    return n

def basic_transformer(network, weight_map, i, ch, x, context):
    H = x.get_output(0).shape[2]
    W = x.get_output(0).shape[3]

    # n c h w -> b (h w) c
    x = network.add_shuffle(x.get_output(0))
    x.first_transpose = trt.Permutation([0, 2, 3, 1])
    x.reshape_dims = (2, -1, ch)

    # attn1
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm1'.format(i))
    
    attn1 = self_attention(network, weight_map, i, ch, n)
    x = network.add_elementwise(attn1.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # attn2
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm2'.format(i))
    attn2 = cross_attention(network, weight_map, i, ch, n, context)
    x = network.add_elementwise(attn2.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # ff
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm3'.format(i))
    ff = feed_forward(network, weight_map, i, ch, n)
    
    x = network.add_elementwise(ff.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # n (h w) c -> n c h w
    x = network.add_shuffle(x.get_output(0))
    x.first_transpose = trt.Permutation([0, 2, 1])
    x.reshape_dims = (2, ch, H, W)
    return x


def spatial_transformer(network, weight_map, i, ch, h, context):
    # return h
    # norm
    n = group_norm(network, weight_map, h, '{}.norm'.format(i), 1e-6)
    # proj_in
    n = conv(network, weight_map, n, ch, '{}.proj_in'.format(i), 1, 0, 1)

    # BasicTransformerBlock
    n = basic_transformer(network, weight_map, i, ch, n, context)

    # proj_out
    n = conv(network, weight_map, n, ch, '{}.proj_out'.format(i), 1, 0, 1)

    h = network.add_elementwise(n.get_output(0), h.get_output(0), trt.ElementWiseOperation.SUM)
    return h

def downsample(network, weight_map, i, ch, x):
    x = conv(network, weight_map, x, ch, '{}.op'.format(i), 3, 1, 2)

    return x

def upsample(network, weight_map, i, ch, x):
    x = network.add_resize(x.get_output(0))
    x.scales = [1, 1, 2, 2]
    x.resize_mode = trt.ResizeMode.NEAREST

    x = conv(network, weight_map, x, ch, '{}.conv'.format(i), 3, 1, 1)

    return x

def input_block(network, weight_map, embed_weight, h, emb, context, model_name):
    hs = []
    h = input_first(network, weight_map, model_name, h)
    h = network.add_slice(h.get_output(0), trt.Dims([0, 0, 0, 0]), trt.Dims([2, 320, 32, 48]), trt.Dims([1, 1, 1, 1]))
    h.mode = trt.SliceMode.WRAP

    #return h
    hs.append(h)

    channel_mult = [1, 2, 4, 4]
    num_res_blocks = [2] * 4

    model_channels = 320
    index = 1
    for level, mult in enumerate(channel_mult):
        ch = model_channels * mult
        for nr in range(num_res_blocks[level]):
            pre = '{}.input_blocks.{}'.format(model_name, index)
            h = resblock(network, weight_map, embed_weight, '{}.0'.format(pre), ch, h, emb)
            print('resblock: ', h.get_output(0).shape)
            if level != len(channel_mult) -1:
                h = spatial_transformer(network, weight_map, '{}.1'.format(pre), ch, h, context)
            hs.append(h)

            # ch = mult * model_channels
            index = index + 1

        if level != len(channel_mult) - 1:
            pre = '{}.input_blocks.{}'.format(model_name, index)
            out_ch = ch
            h = downsample(network, weight_map, '{}.0'.format(pre), out_ch, h)
            hs.append(h)
            index = index + 1
        
        # if index == 10:
    return hs, h

def zero_convs(network, weight_map, x, i):
    ch = x.get_output(0).shape[1]
    x = conv(network, weight_map, x, ch, 'control_model.zero_convs.{}.0'.format(i), 1, 0, 1)
    return x

def input_block_control(network, weight_map, embed_weight, h, emb, context, hint):
    hs = []
    h = input_first(network, weight_map, 'control_model', h)
    h = network.add_elementwise(h.get_output(0), hint, trt.ElementWiseOperation.SUM)

    h = network.add_slice(h.get_output(0), trt.Dims([0, 0, 0, 0]), trt.Dims([2, 320, 32, 48]), trt.Dims([1, 1, 1, 1]))
    h.mode = trt.SliceMode.WRAP
    hs.append(zero_convs(network, weight_map, h, 0))
    # h [2, 320, 32, 48]

    channel_mult = [1, 2, 4, 4]
    num_res_blocks = [2] * 4

    model_channels = 320
    index = 1
    for level, mult in enumerate(channel_mult):
        ch = model_channels * mult
        for nr in range(num_res_blocks[level]):
            pre = 'control_model.input_blocks.{}'.format(index)
            h = resblock(network, weight_map, embed_weight, '{}.0'.format(pre), ch, h, emb)
            print('resblock: ', h.get_output(0).shape)
            if level != len(channel_mult) -1:
                h = spatial_transformer(network, weight_map, '{}.1'.format(pre), ch, h, context)
            hs.append(zero_convs(network, weight_map, h, index))

            # ch = mult * model_channels
            index = index + 1

        if level != len(channel_mult) - 1:
            pre = 'control_model.input_blocks.{}'.format(index)
            out_ch = ch
            h = downsample(network, weight_map, '{}.0'.format(pre), out_ch, h)
            hs.append(zero_convs(network, weight_map, h, index))
            index = index + 1
        
        # if index == 10:
    return hs, h

def middle_block(network, weight_map, embed_weight, h, emb, context, model_name):
    pre = '{}.middle_block'.format(model_name)
    h = resblock(network, weight_map, embed_weight, '{}.0'.format(pre), 1280, h, emb)
    h = spatial_transformer(network, weight_map, '{}.1'.format(pre), 1280, h, context)
    h = resblock(network, weight_map, embed_weight, '{}.2'.format(pre), 1280, h, emb)
    return h

def output_blocks(network, weight_map, embed_weight, h, emb, context, control, hs):
    channel_mult = [1, 2, 4, 4]
    num_res_blocks = [2] * 4

    model_channels = 320
    index = 0
    for level, mult in list(enumerate(channel_mult))[::-1]:
        ch = model_channels * mult
        for i in range(num_res_blocks[level] + 1):
            print(control[-1].shape, hs[-1].shape, len(hs), h.get_output(0).shape)
            c = network.add_elementwise(control.pop(), hs.pop(), trt.ElementWiseOperation.SUM)
            h = network.add_concatenation([h.get_output(0), c.get_output(0)])
            print('output: ', index, h.get_output(0).shape)
            pre = 'model.diffusion_model.output_blocks.{}'.format(index)
            h = resblock(network, weight_map, embed_weight, '{}.0'.format(pre), ch, h, emb)
            print('resblock: ', h.get_output(0).shape)
            if level != len(channel_mult) -1:
                h = spatial_transformer(network, weight_map, '{}.1'.format(pre), ch, h, context)
            
            if level and i == num_res_blocks[level]:
                h = upsample(network, weight_map,
                             '{}.{}'.format(pre, 1 if level == len(channel_mult) - 1 else 2), ch, h)
            index = index + 1
    print(h.get_output(0).shape, len(hs), len(control), index)
    return h

def control_net(network, weight_map, embed_weight, h, hint, emb, context):
    # #####################
    # # time_embed
    # #####################

    #####################
    # input_blocks
    #####################
    control, h = input_block_control(network, weight_map, embed_weight, h, emb, context, hint)
    print(h.get_output(0).shape)

    #####################
    # middle_blocks
    #####################   
    h = middle_block(network, weight_map, embed_weight, h, emb, context, 'control_model')
    h = conv(network, weight_map, h, 1280, 'control_model.middle_block_out.0', 1, 0, 1)

    control.append(h)
    return control


def unet(network, weight_map, embed_weight, h, emb, context, control):
    # #####################
    # # time_embed
    # #####################


    #####################
    # input_blocks
    #####################
    hs, h = input_block(network, weight_map, embed_weight, h, emb, context, 'model.diffusion_model')
    print(h.get_output(0).shape)

    #####################
    # middle_blocks
    #####################   
    h = middle_block(network, weight_map, embed_weight, h, emb, context, 'model.diffusion_model')
    print(h.get_output(0).shape)

    h = network.add_elementwise(h.get_output(0), control.pop().get_output(0), trt.ElementWiseOperation.SUM)

    #####################
    # output_blocks
    #####################
    h = output_blocks(network, weight_map, embed_weight, h, emb, context, control, hs)

    # out
    # group_norm
    # h = group_norm_sile(network, weight_map, h)
    h = group_norm(network, weight_map, h, 'model.diffusion_model.out.0', silu=True)
    # silu
    # h = silu(network, h)
    # conv_nd
    h = conv(network, weight_map, h, 4, 'model.diffusion_model.out.2', 3, 1, 1)

    return h

def create_unet_output_engine(weight_map, embed_weight, context):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    t_emb = network.add_input("t_emb", trt.int32, trt.Dims([1]))
    c = network.add_input("context", trt.float32, trt.Dims([2, 77, 8 * 4560]))

    control = []
    control.append(network.add_input("c1", trt.float16, trt.Dims([2, 320, 32, 48])))
    control.append(network.add_input("c2", trt.float16, trt.Dims([2, 320, 32, 48])))
    control.append(network.add_input("c3", trt.float16, trt.Dims([2, 320, 32, 48])))
    control.append(network.add_input("c4", trt.float16, trt.Dims([2, 320, 16, 24])))
    control.append(network.add_input("c5", trt.float16, trt.Dims([2, 640, 16, 24])))
    control.append(network.add_input("c6", trt.float16, trt.Dims([2, 640, 16, 24])))
    control.append(network.add_input("c7", trt.float16, trt.Dims([2, 640, 8, 12])))
    control.append(network.add_input("c8", trt.float16, trt.Dims([2, 1280, 8, 12])))
    control.append(network.add_input("c9", trt.float16, trt.Dims([2, 1280, 8, 12])))
    control.append(network.add_input("c10", trt.float16, trt.Dims([2, 1280, 4, 6])))
    control.append(network.add_input("c11", trt.float16, trt.Dims([2, 1280, 4, 6])))
    control.append(network.add_input("c12", trt.float16, trt.Dims([2, 1280, 4, 6])))
    control.append(network.add_input("c13", trt.float16, trt.Dims([2, 1280, 4, 6])))

    hs = []
    hs.append(network.add_input("h1", trt.float16, trt.Dims([2, 320, 32, 48])))
    hs.append(network.add_input("h2", trt.float16, trt.Dims([2, 320, 32, 48])))
    hs.append(network.add_input("h3", trt.float16, trt.Dims([2, 320, 32, 48])))
    hs.append(network.add_input("h4", trt.float16, trt.Dims([2, 320, 16, 24])))
    hs.append(network.add_input("h5", trt.float16, trt.Dims([2, 640, 16, 24])))
    hs.append(network.add_input("h6", trt.float16, trt.Dims([2, 640, 16, 24])))
    hs.append(network.add_input("h7", trt.float16, trt.Dims([2, 640, 8, 12])))
    hs.append(network.add_input("h8", trt.float16, trt.Dims([2, 1280, 8, 12])))
    hs.append(network.add_input("h9", trt.float16, trt.Dims([2, 1280, 8, 12])))
    hs.append(network.add_input("h10", trt.float16, trt.Dims([2, 1280, 4, 6])))
    hs.append(network.add_input("h11", trt.float16, trt.Dims([2, 1280, 4, 6])))    
    hs.append(network.add_input("h12", trt.float16, trt.Dims([2, 1280, 4, 6]))) 
    hs.append(network.add_input("h13", trt.float16, trt.Dims([2, 1280, 4, 6])))

    for i in range(len(hs)):
        hs[i].dtype = trt.DataType.HALF
        hs[i].allowed_formats = 1 << int(trt.TensorFormat.HWC8)

        control[i].dtype = trt.DataType.HALF
        control[i].allowed_formats = 1 << int(trt.TensorFormat.HWC8)

    context['context'] = c

    h = network.add_elementwise(hs.pop(), control.pop(), trt.ElementWiseOperation.SUM)
    #####################
    # output_blocks
    #####################
    x = output_blocks(network, weight_map, embed_weight, h, t_emb, context, control, hs)

    # out
    # group_norm
    x = group_norm(network, weight_map, x, 'model.diffusion_model.out.0', silu=True)
    # silu
    # x = silu(network, x)
    # conv_nd
    x = conv(network, weight_map, x, 4, 'model.diffusion_model.out.2', 3, 1, 1)

    x.get_output(0).name = 'out'
    network.mark_output(x.get_output(0))

    # builder.max_batch_size = 1
    config.max_workspace_size = 1<<30
    config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)

    del network

    return engine

def create_unet_input_engine(weight_map, embed_weight, context):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    h = network.add_input("x", trt.float32, trt.Dims([1, 4, 32, 48]))
    t_emb = network.add_input("t_emb", trt.int32, trt.Dims([1]))
    c = network.add_input("context", trt.float32, trt.Dims([2, 77, 8 * 4560]))

    context['context'] = c
    #####################
    # input_blocks
    #####################
    hs, h = input_block(network, weight_map, embed_weight, h, t_emb, context, 'model.diffusion_model')
    print(h.get_output(0).shape)

    #####################
    # middle_blocks
    #####################   
    h = middle_block(network, weight_map, embed_weight, h, t_emb, context, 'model.diffusion_model')
    print(h.get_output(0).shape)
    hs.append(h)

    for i in range(len(hs)):
        hs[i].get_output(0).name = 'h{}'.format(i + 1)
        hs[i].get_output(0).dtype = trt.DataType.HALF
        hs[i].get_output(0).allowed_formats = 1 << int(trt.TensorFormat.HWC8)
        network.mark_output(hs[i].get_output(0))
    
    # builder.max_batch_size = 1
    config.max_workspace_size = 1<<30
    config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)

    del network

    return engine

def create_control_engine(weight_map, embed_weight, context):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    h = network.add_input("x", trt.float32, trt.Dims([1, 4, 32, 48]))
    hint = network.add_input("hint", trt.float32, trt.Dims([1, 320, 32, 48]))
    t_emb = network.add_input("t_emb", trt.int32, trt.Dims([1]))
    c = network.add_input("context", trt.float32, trt.Dims([2, 77, 8 * 4560]))

    context['context'] = c
    control = control_net(network, weight_map, embed_weight, h, hint, t_emb, context)

    for i in range(len(control)):
        control[i].get_output(0).name = 'c{}'.format(i + 1)
        control[i].get_output(0).dtype = trt.DataType.HALF
        control[i].get_output(0).allowed_formats = 1 << int(trt.TensorFormat.HWC8)
        network.mark_output(control[i].get_output(0))
    
    # builder.max_batch_size = 1
    config.max_workspace_size = 1<<30
    config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)

    del network

    return engine

def convert_trt_engine(input_model_path):
    # """
    weight_map = load_torch_model(input_model_path)
    embed_weight = compute_embedding()
    context = {'start' : 0}

    engine = create_control_engine(weight_map, embed_weight, context)
    engine_data = bytes(engine.serialize())
    with open('./trt/sd_control_fp16.plan', 'wb') as f:
        f.write(engine_data)

    context['start'] = 1440
    engine = create_unet_input_engine(weight_map, embed_weight, context)
    engine_data = bytes(engine.serialize())
    with open('./trt/sd_unet_input_fp16.plan', 'wb') as f:
        f.write(engine_data)

    context['start'] = 2880
    engine = create_unet_output_engine(weight_map, embed_weight, context)
    engine_data = bytes(engine.serialize())
    with open('./trt/sd_unet_output_fp16.plan', 'wb') as f:
        f.write(engine_data)

    # """

convert_trt_engine('/home/player/ControlNet/models/control_sd15_canny.pth')
