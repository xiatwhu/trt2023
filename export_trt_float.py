import os
import sys

import numpy as np
import tensorrt as trt
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding

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

    del model
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
    # if ch < 1280:
    #     x.precision = trt.DataType.FLOAT
    return x

def input_first(network, weight_map, pre, h):
    h = conv(network, weight_map, h, 320, '{}.input_blocks.0.0'.format(pre), 3, 1, 1)
    h.precision = trt.DataType.FLOAT
    return h

def group_norm(network, weight_map, h, pre, epsilon=EPS):
    N = h.get_output(0).shape[0]
    H = h.get_output(0).shape[2]
    W = h.get_output(0).shape[3]

    h = network.add_shuffle(h.get_output(0))
    h.reshape_dims = (N, 32, -1, W)

    # instance norm
    s = network.add_constant([1, 32, 1, 1], np.full((32), 1, dtype=np.float32))
    b = network.add_constant([1, 32, 1, 1], np.full((32), 0, dtype=np.float32))
    n = network.add_normalization(
        h.get_output(0),
        scale=s.get_output(0),
        bias=b.get_output(0),
        axesMask=1 << 2 | 1 << 3)
    n.epsilon = epsilon
    # n.compute_precision = trt.float16
    
    n = network.add_shuffle(n.get_output(0))
    n.reshape_dims = (N, -1, H, W)

    n = network.add_scale(n.get_output(0), trt.ScaleMode.CHANNEL,
                          shift=weight_map['{}.bias'.format(pre)],
                          scale=weight_map['{}.weight'.format(pre)])
    print('group_norm: ', n.get_output(0).shape)

    return n

def layer_norm(network, weight_map, h, pre, epsilon=EPS):
    scale_np = weight_map['{}.weight'.format(pre)]
    ch = scale_np.shape[0]
    scale = network.add_constant([1, ch, 1, 1], scale_np)
    bias_np = weight_map['{}.bias'.format(pre)]
    bias = network.add_constant([1, ch, 1, 1], bias_np)
    n = network.add_normalization(
        h.get_output(0),
        scale=scale.get_output(0),
        bias=bias.get_output(0),
        axesMask=1 << 1)
    assert n
    n.epsilon = epsilon

    return n    

def resblock(network, weight_map, embed_weight, i, ch, h, emb):
    ## in_layers
    # group_norm
    n = group_norm(network, weight_map, h, '{}.in_layers.0'.format(i))
    # silu
    n = silu(network, n)
    # conv_nd
    n = conv(network, weight_map, n, ch, '{}.in_layers.2'.format(i), 3, 1, 1)

    print('in_layers: ', n.get_output(0).shape)

    ## emb_layers
    # # silu
    # m = silu(network, emb)
    # # linear
    # m = network.add_fully_connected(
    #         m.get_output(0),
    #         num_outputs=ch,
    #         kernel=weight_map['{}.emb_layers.1.weight'.format(i)],
    #         bias=weight_map['{}.emb_layers.1.bias'.format(i)])
    # print('emb_layers: ', m.get_output(0).shape)
    m = network.add_constant([20, ch, 1, 1], embed_weight.pop(0))
    m = network.add_gather(m.get_output(0), emb, axis=0)
    print('emb_layers: ', m.get_output(0).shape)

    n = network.add_elementwise(n.get_output(0), m.get_output(0), trt.ElementWiseOperation.SUM)

    ## out_layers
    n = group_norm(network, weight_map, n, '{}.out_layers.0'.format(i))
    n = silu(network, n)
    n = conv(network, weight_map, n, ch, '{}.out_layers.3'.format(i), 3, 1, 1)

    print('out_layers: ', n.get_output(0).shape)

    in_ch = h.get_output(0).shape[1]
    if in_ch != ch:
        # skip_connection
        h = conv(network, weight_map, h, ch, '{}.skip_connection'.format(i), 1, 0, 1)

    h = network.add_elementwise(n.get_output(0), h.get_output(0), trt.ElementWiseOperation.SUM)
    return h

def self_attention(network, weight_map, i, ch, x):
    h = x.get_output(0).shape[2]
    w = x.get_output(0).shape[3]
    
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5

    q = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn1.to_q.weight'.format(i)])
    q.padding = (0, 0)
    q.stride = (1, 1)

    k = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn1.to_k.weight'.format(i)])
    k.padding = (0, 0)
    k.stride = (1, 1)

    v = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn1.to_v.weight'.format(i)])
    v.padding = (0, 0)
    v.stride = (1, 1)

    # q [2, c, h, w] -> [2 * head, h*w, d]
    # k [2, c, h, w] -> [2 * head, h*w, d]
    # v [2, c, h, w] -> [2 * head, h*w, d]
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (16, ch // 8, -1)
    q.second_transpose = trt.Permutation([0, 2, 1])

    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (16, ch // 8, -1)
    k.second_transpose = trt.Permutation([0, 2, 1])

    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (16, ch // 8, -1)
    v.second_transpose = trt.Permutation([0, 2, 1])
    print(q.get_output(0).shape)
    print(k.get_output(0).shape)

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'bik,bjk->bij')
    print(s.get_output(0).shape)
    s = network.add_scale(s.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([scale], np.float32)))
    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'bij,bjk->bik')
    # [2 * head, h * w, d] -> [2, c, h, w]
    out = network.add_shuffle(out.get_output(0))
    out.first_transpose = trt.Permutation([0, 2, 1])
    out.reshape_dims = (2, ch, h, w)

    # to_out
    out = conv(network, weight_map, out, ch, '{}.transformer_blocks.0.attn1.to_out.0'.format(i), 1, 0, 1)

    return out

def cross_attention(network, weight_map, i, ch, x, context):
    h = x.get_output(0).shape[2]
    w = x.get_output(0).shape[3]
    
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5

    q = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn2.to_q.weight'.format(i)])
    q.padding = (0, 0)
    q.stride = (1, 1)

    # k = network.add_convolution(
    #         input=context,
    #         num_output_maps=ch,
    #         kernel_shape=(1, 1),
    #         kernel=weight_map['{}.transformer_blocks.0.attn2.to_k.weight'.format(i)])
    # k.padding = (0, 0)
    # k.stride = (1, 1)

    # v = network.add_convolution(
    #         input=context,
    #         num_output_maps=ch,
    #         kernel_shape=(1, 1),
    #         kernel=weight_map['{}.transformer_blocks.0.attn2.to_v.weight'.format(i)])
    # v.padding = (0, 0)
    # v.stride = (1, 1)

    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (16, ch // 8, -1)
    q.second_transpose = trt.Permutation([0, 2, 1])

    dim = ch // 8
    k = network.add_slice(context['context'],
                          trt.Dims([0, 0, context['start']]),
                          trt.Dims([16, 77, dim]),
                          trt.Dims([1, 1, 1]))
    v = network.add_slice(context['context'],
                          trt.Dims([0, 0, context['start'] + dim]),
                          trt.Dims([16, 77, dim]),
                          trt.Dims([1, 1, 1]))
    context['start'] += 2 * dim
    print('start: ', context['start'])
    print('cross_attn: ', q.get_output(0).shape, k.get_output(0).shape, v.get_output(0).shape)

    # k = network.add_shuffle(k.get_output(0))
    # k.reshape_dims = (16, ch // 8, -1)
    # k.second_transpose = trt.Permutation([0, 2, 1])

    # v = network.add_shuffle(v.get_output(0))
    # v.reshape_dims = (16, ch // 8, -1)
    # v.second_transpose = trt.Permutation([0, 2, 1])

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'bik,bjk->bij')
    print(s.get_output(0).shape)
    s = network.add_scale(s.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([scale], np.float32)))
    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'bij,bjk->bik')
    out = network.add_shuffle(out.get_output(0))
    out.first_transpose = trt.Permutation([0, 2, 1])
    out.reshape_dims = (2, ch, h, w)

    # to_out
    out = conv(network, weight_map, out, ch, '{}.transformer_blocks.0.attn2.to_out.0'.format(i), 1, 0, 1)

    return out

def feed_forward(network, weight_map, i, ch, x):
    n = conv(network, weight_map, x, ch * 8, '{}.transformer_blocks.0.ff.net.0.proj'.format(i), 1, 0, 1)

    h = n.get_output(0).shape[2]
    w = n.get_output(0).shape[3]
    n1 = network.add_slice(n.get_output(0), trt.Dims([0, 0, 0, 0]), trt.Dims([2, ch * 4, h, w]), trt.Dims([1, 1, 1, 1]))
    n2 = network.add_slice(n.get_output(0), trt.Dims([0, ch * 4, 0, 0]), trt.Dims([2, ch * 4, h, w]), trt.Dims([1, 1, 1, 1]))

    # gelu
    e = network.add_scale(n2.get_output(0), mode=trt.ScaleMode.UNIFORM, scale=trt.Weights(np.array([2 ** -0.5], np.float32)))
    e = network.add_unary(e.get_output(0), trt.UnaryOperation.ERF)
    e = network.add_scale(e.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([0.5], np.float32)),
                          shift=trt.Weights(np.array([0.5], np.float32)))

    n = network.add_elementwise(n2.get_output(0), e.get_output(0), trt.ElementWiseOperation.PROD)
    n = network.add_elementwise(n.get_output(0), n1.get_output(0), trt.ElementWiseOperation.PROD)

    n = conv(network, weight_map, n, ch, '{}.transformer_blocks.0.ff.net.2'.format(i), 1, 0, 1)

    return n

def basic_transformer(network, weight_map, i, ch, x, context):
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
            print(control[-1].get_output(0).shape, hs[-1].get_output(0).shape, len(hs), h.get_output(0).shape)
            c = network.add_elementwise(control.pop().get_output(0), hs.pop().get_output(0), trt.ElementWiseOperation.SUM)
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
    h = group_norm(network, weight_map, h, 'model.diffusion_model.out.0')
    # silu
    h = silu(network, h)
    # conv_nd
    h = conv(network, weight_map, h, 4, 'model.diffusion_model.out.2', 3, 1, 1)

    return h

def create_df_engine(weight_map, embed_weight):
    # logger = trt.Logger(trt.Logger.VERBOSE)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    h = network.add_input("x", trt.float32, trt.Dims([1, 4, 32, 48]))
    hint = network.add_input("hint", trt.float32, trt.Dims([1, 320, 32, 48]))
    t_emb = network.add_input("t_emb", trt.int32, trt.Dims([1]))
    context = network.add_input("context", trt.float32, trt.Dims([16, 77, 4560]))

    context = {'context': context, 'start': 0}
    control = control_net(network, weight_map, embed_weight, h, hint, t_emb, context)
    x = unet(network, weight_map, embed_weight, h, t_emb, context, control)

    print(x.get_output(0).shape)

    x.get_output(0).name = 'out'
    network.mark_output(x.get_output(0))

    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     if layer.type != trt.LayerType.CONVOLUTION:
    #         network.get_layer(i).precision = trt.DataType.FLOAT
    
    # builder.max_batch_size = 1
    config.max_workspace_size = 2<<30
    # config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    engine = builder.build_engine(network, config)

    del network

    return engine


def convert_trt_engine(input_model_path):
    weight_map = load_torch_model(input_model_path)
    embed_weight = compute_embedding()
    df_engine = create_df_engine(weight_map, embed_weight)
    df_data = bytes(df_engine.serialize())
    with open('./df_float.plan', 'wb') as f:
        f.write(df_data)

    del weight_map

convert_trt_engine('/home/player/ControlNet/models/control_sd15_canny.pth')
