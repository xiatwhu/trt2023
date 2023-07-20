import os
import sys

import numpy as np
import tensorrt as trt
import torch

EPS = 1e-5

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

def input_first(network, weight_map, i, h, emb, context):
    h = network.add_convolution(
            # input=h.get_output(0),
            input=h,
            num_output_maps=320,
            kernel_shape=(3, 3),
            kernel=weight_map['model.diffusion_model.input_blocks.0.0.weight'],
            bias=weight_map['model.diffusion_model.input_blocks.0.0.bias'])
    h.padding = (1, 1)
    h.stride = (1, 1)
    assert h
    return h

# def input_downsample(network, weight_map, i, h, emb, context):
#     h = network.add_convolution(
#             input=h.get_output(0),
#             num_output_maps=)

def group_norm(network, weight_map, h, pre, epsilon=EPS):
    H = h.get_output(0).shape[2]
    W = h.get_output(0).shape[3]

    h = network.add_shuffle(h.get_output(0))
    h.reshape_dims = (1, 32, -1, W)

    # instance norm
    s = network.add_constant([1, 32, 1, 1], np.full((32), 1, dtype=np.float32))
    b = network.add_constant([1, 32, 1, 1], np.full((32), 0, dtype=np.float32))
    n = network.add_normalization(
        h.get_output(0),
        scale=s.get_output(0),
        bias=b.get_output(0),
        axesMask=1 << 2 | 1 << 3)
    n.epsilon = epsilon
    #n.compute_precision = trt.float16
    
    n = network.add_shuffle(n.get_output(0))
    n.reshape_dims = (1, -1, H, W)

    n = network.add_scale(n.get_output(0), trt.ScaleMode.CHANNEL,
                          shift=weight_map['{}.bias'.format(pre)],
                          scale=weight_map['{}.weight'.format(pre)])
    print('group_norm: ', n.get_output(0).shape)

    # scale_np = weight_map['{}.weight'.format(pre)]
    # ch = scale_np.shape[0]
    # scale = network.add_constant([1, ch, 1, 1], scale_np)
    # bias_np = weight_map['{}.bias'.format(pre)]
    # bias = network.add_constant([1, ch, 1, 1], bias_np)
    # print("\n\n\n\n")
    # print(h.get_output(0).shape)
    # print(scale.get_output(0).shape)
    # print(bias.get_output(0).shape)
    
    # n = network.add_normalization(
    #     h.get_output(0),
    #     scale=scale.get_output(0),
    #     bias=bias.get_output(0),
    #     axesMask=1 << 2 | 1 << 3)
    # print(n.name)
    # print("\n\n\n\n")
    # assert n
    # n.epsilon = epsilon
    # n.num_groups = 32
    #n.axes = 1 << 3
    #n.compute_precision = trt.float16

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
    #n.num_groups = 32
    #n.compute_precision = trt.float16

    return n    

def resblock(network, weight_map, i, ch, h, emb, context):
    ## in_layers
    # group_norm
    n = group_norm(network, weight_map, h, '{}.in_layers.0'.format(i))
    # silu
    n = silu(network, n)
    # conv_nd
    n = network.add_convolution(
            input=n.get_output(0),
            num_output_maps=ch,
            kernel_shape=(3, 3),
            kernel=weight_map['{}.in_layers.2.weight'.format(i)],
            bias=weight_map['{}.in_layers.2.bias'.format(i)])
    assert n
    n.padding = (1, 1)
    n.stride = (1, 1)
    print('in_layers: ', n.get_output(0).shape)

    ## emb_layers
    # silu
    m = silu(network, emb)
    # linear
    m = network.add_fully_connected(
            m.get_output(0),
            num_outputs=ch,
            kernel=weight_map['{}.emb_layers.1.weight'.format(i)],
            bias=weight_map['{}.emb_layers.1.bias'.format(i)])
    print('emb_layers: ', m.get_output(0).shape)

    n = network.add_elementwise(n.get_output(0), m.get_output(0), trt.ElementWiseOperation.SUM)

    ## out_layers
    n = group_norm(network, weight_map, n, '{}.out_layers.0'.format(i))
    n = silu(network, n)
    n = network.add_convolution(
            input=n.get_output(0),
            num_output_maps=ch,
            kernel_shape=(3, 3),
            kernel=weight_map['{}.out_layers.3.weight'.format(i)],
            bias=weight_map['{}.out_layers.3.bias'.format(i)])
    assert n
    n.padding = (1, 1)
    n.stride = (1, 1)
    print('out_layers: ', n.get_output(0).shape)

    in_ch = h.get_output(0).shape[1]
    if in_ch != ch:
        # skip_connection
        h = network.add_convolution(
                input=h.get_output(0),
                num_output_maps=ch,
                kernel_shape=(1, 1),
                kernel=weight_map['{}.skip_connection.weight'.format(i)],
                bias=weight_map['{}.skip_connection.bias'.format(i)])
        h.padding = (0, 0)
        h.stride = (1, 1)

    h = network.add_elementwise(n.get_output(0), h.get_output(0), trt.ElementWiseOperation.SUM)
    return h

def self_attention(network, weight_map, i, ch, x, emb, context):
    h = x.get_output(0).shape[2]
    w = x.get_output(0).shape[3]
    
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5
    q = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn1.to_q.weight'.format(i)] * scale)
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

    # q [c, h*w] -> [head * d, h*w]
    # k [c, h*w] -> [head * d, h*w]
    # v [c, h*w] -> [head * d, h*w]
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (8, ch // 8, -1)

    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (8, ch // 8, -1)

    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (8, ch // 8, -1)
    print(q.get_output(0).shape)
    print(k.get_output(0).shape)

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'bki,bkj->bij')
    print(s.get_output(0).shape)
    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'bij,bkj->bki')
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, ch, h, w)

    # to_out
    out = network.add_convolution(
            input=out.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn1.to_out.0.weight'.format(i)],
            bias=weight_map['{}.transformer_blocks.0.attn1.to_out.0.bias'.format(i)])
    out.padding = (0, 0)
    out.stride = (1, 1)
    return out

def cross_attention(network, weight_map, i, ch, x, emb, context):
    h = x.get_output(0).shape[2]
    w = x.get_output(0).shape[3]
    
    heads = 8
    dim_head = ch / heads
    scale = dim_head ** -0.5
    q = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn2.to_q.weight'.format(i)] * scale)
    q.padding = (0, 0)
    q.stride = (1, 1)

    k = network.add_convolution(
            input=context,
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn2.to_k.weight'.format(i)])
    k.padding = (0, 0)
    k.stride = (1, 1)

    v = network.add_convolution(
            input=context,
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn2.to_v.weight'.format(i)])
    v.padding = (0, 0)
    v.stride = (1, 1)

    # q [c, h*w] -> [head * d, h*w]
    # k [c, h*w] -> [head * d, h*w]
    # v [c, h*w] -> [head * d, h*w]
    q = network.add_shuffle(q.get_output(0))
    q.reshape_dims = (8, ch // 8, -1)

    k = network.add_shuffle(k.get_output(0))
    k.reshape_dims = (8, ch // 8, -1)

    v = network.add_shuffle(v.get_output(0))
    v.reshape_dims = (8, ch // 8, -1)
    print(q.get_output(0).shape)
    print(k.get_output(0).shape)

    s = network.add_einsum([q.get_output(0), k.get_output(0)], 'bki,bkj->bij')
    print(s.get_output(0).shape)
    s = network.add_softmax(s.get_output(0))
    s.axes = 1<<2

    out = network.add_einsum([s.get_output(0), v.get_output(0)], 'bij,bkj->bki')
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, ch, h, w)

    # to_out
    out = network.add_convolution(
            input=out.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.attn2.to_out.0.weight'.format(i)],
            bias=weight_map['{}.transformer_blocks.0.attn2.to_out.0.bias'.format(i)])
    out.padding = (0, 0)
    out.stride = (1, 1)
    return out

def feed_forward(network, weight_map, i, ch, x):
    n = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch*8,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.ff.net.0.proj.weight'.format(i)],
            bias=weight_map['{}.transformer_blocks.0.ff.net.0.proj.bias'.format(i)])
    n.padding = (0, 0)
    n.stride = (1, 1)

    h = n.get_output(0).shape[2]
    w = n.get_output(0).shape[3]
    n1 = network.add_slice(n.get_output(0), trt.Dims([0, 0, 0, 0]), trt.Dims([1, ch * 4, h, w]), trt.Dims([1, 1, 1, 1]))
    n2 = network.add_slice(n.get_output(0), trt.Dims([0, ch * 4, 0, 0]), trt.Dims([1, ch * 4, h, w]), trt.Dims([1, 1, 1, 1]))

    # gelu
    e = network.add_scale(n2.get_output(0), mode=trt.ScaleMode.UNIFORM, scale=trt.Weights(np.array([2 ** -0.5], np.float32)))
    e = network.add_unary(e.get_output(0), trt.UnaryOperation.ERF)
    e = network.add_scale(e.get_output(0), mode=trt.ScaleMode.UNIFORM,
                          scale=trt.Weights(np.array([0.5], np.float32)),
                          shift=trt.Weights(np.array([0.5], np.float32)))

    n = network.add_elementwise(n2.get_output(0), e.get_output(0), trt.ElementWiseOperation.PROD)
    n = network.add_elementwise(n.get_output(0), n1.get_output(0), trt.ElementWiseOperation.PROD)

    n = network.add_convolution(
            input=n.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.transformer_blocks.0.ff.net.2.weight'.format(i)],
            bias=weight_map['{}.transformer_blocks.0.ff.net.2.bias'.format(i)])
    n.padding = (0, 0)
    n.stride = (1, 1)
    return n

def basic_transformer(network, weight_map, i, ch, x, emb, context):
    # attn1
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm1'.format(i))
    attn1 = self_attention(network, weight_map, i, ch, n, emb, context)
    x = network.add_elementwise(attn1.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # attn2
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm2'.format(i))
    attn2 = cross_attention(network, weight_map, i, ch, n, emb, context)
    x = network.add_elementwise(attn2.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # ff
    n = layer_norm(network, weight_map, x, '{}.transformer_blocks.0.norm3'.format(i))
    ff = feed_forward(network, weight_map, i, ch, n)
    
    x = network.add_elementwise(ff.get_output(0), x.get_output(0), trt.ElementWiseOperation.SUM)

    # TODO
    return x


def spatial_transformer(network, weight_map, i, ch, h, emb, context):
    # return h
    # norm
    n = group_norm(network, weight_map, h, '{}.norm'.format(i), 1e-6)
    # proj_in
    n = network.add_convolution(
            input=n.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.proj_in.weight'.format(i)],
            bias=weight_map['{}.proj_in.bias'.format(i)])
    assert n
    n.padding = (0, 0)
    n.stride = (1, 1)

    # BasicTransformerBlock
    n = basic_transformer(network, weight_map, i, ch, n, emb, context)

    # proj_out
    n = network.add_convolution(
            input=n.get_output(0),
            num_output_maps=ch,
            kernel_shape=(1, 1),
            kernel=weight_map['{}.proj_out.weight'.format(i)],
            bias=weight_map['{}.proj_out.bias'.format(i)])

    h = network.add_elementwise(n.get_output(0), h.get_output(0), trt.ElementWiseOperation.SUM)
    return h

def downsample(network, weight_map, i, ch, x):
    x = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(3, 3),
            kernel=weight_map['{}.op.weight'.format(i)],
            bias=weight_map['{}.op.bias'.format(i)])
    x.padding = (1, 1)
    x.stride = (2, 2)
    return x

def upsample(network, weight_map, i, ch, x):
    x = network.add_resize(x.get_output(0))
    x.scales = [1, 1, 2, 2]
    x.resize_mode = trt.ResizeMode.NEAREST

    x = network.add_convolution(
            input=x.get_output(0),
            num_output_maps=ch,
            kernel_shape=(3, 3),
            kernel=weight_map['{}.conv.weight'.format(i)],
            bias=weight_map['{}.conv.bias'.format(i)])
    x.padding = (1, 1)
    x.stride = (1, 1)
    return x

def input_block(network, weight_map, h, emb, context):
    hs = []
    h = input_first(network, weight_map, 0, h, emb, context)
    #return h
    hs.append(h)

    channel_mult = [1, 2, 4, 4]
    num_res_blocks = [2] * 4

    model_channels = 320
    index = 1
    for level, mult in enumerate(channel_mult):
        ch = model_channels * mult
        for nr in range(num_res_blocks[level]):
            pre = 'model.diffusion_model.input_blocks.{}'.format(index)
            h = resblock(network, weight_map, '{}.0'.format(pre), ch, h, emb, context)
            print('resblock: ', h.get_output(0).shape)
            if level != len(channel_mult) -1:
                h = spatial_transformer(network, weight_map, '{}.1'.format(pre), ch, h, emb, context)
            hs.append(h)

            # ch = mult * model_channels
            index = index + 1

        if level != len(channel_mult) - 1:
            pre = 'model.diffusion_model.input_blocks.{}'.format(index)
            out_ch = ch
            h = downsample(network, weight_map, '{}.0'.format(pre), out_ch, h)
            hs.append(h)
            index = index + 1
        
        # if index == 10:
    return hs, h

def middle_block(network, weight_map, h, emb, context):
    pre = 'model.diffusion_model.middle_block'
    h = resblock(network, weight_map, '{}.0'.format(pre), 1280, h, emb, context)
    h = spatial_transformer(network, weight_map, '{}.1'.format(pre), 1280, h, emb, context)
    h = resblock(network, weight_map, '{}.2'.format(pre), 1280, h, emb, context)
    return h

def output_blocks(network, weight_map, h, emb, context, control, hs):
    channel_mult = [1, 2, 4, 4]
    num_res_blocks = [2] * 4

    model_channels = 320
    index = 0
    for level, mult in list(enumerate(channel_mult))[::-1]:
        ch = model_channels * mult
        for i in range(num_res_blocks[level] + 1):
            print(control[-1].shape, hs[-1].get_output(0).shape, len(hs), h.get_output(0).shape)
            c = network.add_elementwise(control.pop(), hs.pop().get_output(0), trt.ElementWiseOperation.SUM)
            h = network.add_concatenation([h.get_output(0), c.get_output(0)])
            print('output: ', index, h.get_output(0).shape)
            pre = 'model.diffusion_model.output_blocks.{}'.format(index)
            h = resblock(network, weight_map, '{}.0'.format(pre), ch, h, emb, context)
            print('resblock: ', h.get_output(0).shape)
            if level != len(channel_mult) -1:
                h = spatial_transformer(network, weight_map, '{}.1'.format(pre), ch, h, emb, context)
            
            if level and i == num_res_blocks[level]:
                h = upsample(network, weight_map,
                             '{}.{}'.format(pre, 1 if level == len(channel_mult) - 1 else 2), ch, h)
            index = index + 1
    print(h.get_output(0).shape, len(hs), len(control), index)
    return h

def create_df_engine(weight_map):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    h = network.add_input("x", trt.float32, trt.Dims([1, 4, 32, 48]))
    t_emb = network.add_input("t_emb", trt.float32, trt.Dims([1, 320, 1, 1]))
    context = network.add_input("context", trt.float32, trt.Dims([1, 768, 1, 77]))
    control = []
    control.append(network.add_input("c1", trt.float32, trt.Dims([1, 320, 32, 48])))
    control.append(network.add_input("c2", trt.float32, trt.Dims([1, 320, 32, 48])))
    control.append(network.add_input("c3", trt.float32, trt.Dims([1, 320, 32, 48])))
    control.append(network.add_input("c4", trt.float32, trt.Dims([1, 320, 16, 24])))
    control.append(network.add_input("c5", trt.float32, trt.Dims([1, 640, 16, 24])))
    control.append(network.add_input("c6", trt.float32, trt.Dims([1, 640, 16, 24])))
    control.append(network.add_input("c7", trt.float32, trt.Dims([1, 640, 8, 12])))
    control.append(network.add_input("c8", trt.float32, trt.Dims([1, 1280, 8, 12])))
    control.append(network.add_input("c9", trt.float32, trt.Dims([1, 1280, 8, 12])))
    control.append(network.add_input("c10", trt.float32, trt.Dims([1, 1280, 4, 6])))
    control.append(network.add_input("c11", trt.float32, trt.Dims([1, 1280, 4, 6])))    
    control.append(network.add_input("c12", trt.float32, trt.Dims([1, 1280, 4, 6]))) 
    control.append(network.add_input("c13", trt.float32, trt.Dims([1, 1280, 4, 6]))) 

    #####################
    # time_embed
    #####################
    t = network.add_fully_connected(
                input=t_emb,
                num_outputs=1280,
                kernel=weight_map['model.diffusion_model.time_embed.0.weight'],
                bias=weight_map['model.diffusion_model.time_embed.0.bias'])
    t = silu(network, t)
    emb = network.add_fully_connected(
                input=t.get_output(0),
                num_outputs=1280,
                kernel=weight_map['model.diffusion_model.time_embed.2.weight'],
                bias=weight_map['model.diffusion_model.time_embed.2.bias'])
    # emb [1, 1280, 1, 1]

    #####################
    # input_blocks
    #####################
    hs, h = input_block(network, weight_map, h, emb, context)
    print(h.get_output(0).shape)

    #####################
    # middle_blocks
    #####################   
    h = middle_block(network, weight_map, h, emb, context)
    print(h.get_output(0).shape)

    h = network.add_elementwise(h.get_output(0), control.pop(), trt.ElementWiseOperation.SUM)

    #####################
    # output_blocks
    #####################
    h = output_blocks(network, weight_map, h, emb, context, control, hs)

    # out
    # group_norm
    h = group_norm(network, weight_map, h, 'model.diffusion_model.out.0')
    # silu
    h = silu(network, h)
    # conv_nd
    h = network.add_convolution(
            input=h.get_output(0),
            num_output_maps=4,
            kernel_shape=(3, 3),
            kernel=weight_map['model.diffusion_model.out.2.weight'],
            bias=weight_map['model.diffusion_model.out.2.bias'])
    assert h
    h.padding = (1, 1)
    h.stride = (1, 1)

    h.get_output(0).name = 'out'
    network.mark_output(h.get_output(0))
    # builder.max_batch_size = 1
    config.max_workspace_size = 2<<30
    config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_engine(network, config)

    del network

    return engine


def convert_trt_engine(input_model_path):
    weight_map = load_torch_model(input_model_path)
    df_engine = create_df_engine(weight_map)
    df_data = bytes(df_engine.serialize())
    with open('./df.plan', 'wb') as f:
        f.write(df_data)

    del weight_map

convert_trt_engine('/home/player/ControlNet/models/control_sd15_canny.pth')
