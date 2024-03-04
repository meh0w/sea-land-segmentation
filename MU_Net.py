from tensorflow.keras.layers import GlobalAveragePooling2D, DepthwiseConv2D, Permute, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D, ReLU, Softmax, BatchNormalization, LayerNormalization
from tensorflow.keras import Sequential, optimizers, Model, initializers, activations
from tensorflow import unstack, random, name_scope, concat, transpose, sqrt, zeros, range, stack, meshgrid, reduce_sum, reshape, gather, expand_dims

def conv_block(x_in, filters):
    conv = Conv2D(filters, 3, strides=(1,1), padding='same')(x_in)
    bnorm = BatchNormalization()(conv)
    x_out = ReLU()(bnorm)
    return x_out

def window_partition(x_in, win_size=8):
    B, H, W, C = x_in.shape

    x_in = reshape(x_in,(B, H//win_size, win_size, W//win_size, win_size, C))
    windows = reshape(reshape(transpose(x_in, (0, 1, 3, 2, 4, 5)), (-1, win_size, win_size, C)),(-1, win_size*win_size, C))
    return windows

def window_partition2(x_in, win_size=(8, 8)):

    B, C, H, W = x_in.shape
    x_in = reshape(x_in, (B, C, H // win_size[0], win_size[0],W // win_size[1], win_size[1]))
    windows = reshape(transpose(x_in, perm=[0, 2, 4, 3, 5, 1]), (-1, win_size[0] * win_size[1], C))
    return windows

def window_reverse(x_in, win_size, H=64, W=64, C=512):
    B = int(x_in.shape[0] / (H * W / win_size / win_size))

    x_in = reshape(x_in, (B, H // win_size, W // win_size, win_size, win_size, -1))
    x_out = reshape(transpose(x_in, perm=[0, 1, 3, 2, 4, 5]), (B, H, W, -1))

    return x_out

def window_reverse2(x_in, win_size=(8, 8), H=64, W=64, C=512):
    x_in = reshape(x_in, (-1, H // win_size[0], W // win_size[1],win_size[0], win_size[1], C))
    x_out = reshape(transpose(x_in, perm=[0, 5, 1, 3, 2, 4]), (-1, C, H, W))
    return x_out

def get_bias(win_size=(8,8), num_heads=8):
    coords_h = range(win_size[0])
    coords_w = range(win_size[1])

    coords = stack(meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = Flatten()(coords)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = transpose(relative_coords, perm=[1,2,0])
    a = relative_coords[:, :, 0]
    b = relative_coords[:, :, 1]
    a += win_size[0] - 1
    b += win_size[1] - 1
    a *= 2 * win_size[1] - 1
    relative_coords = stack([a,b])
    relative_position_index = reduce_sum(relative_coords, 0)
    initializer = initializers.TruncatedNormal(stddev=0.02)
    values = initializer(((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))

    bias = transpose(reshape(gather(values, reshape(relative_position_index, -1)), (win_size[0] * win_size[1], win_size[0] * win_size[1], -1)), perm=[2,0,1]) 
    
    return expand_dims(bias, axis=0)

def dw_conv_block(x_in):
    conv = DepthwiseConv2D((3,3))(x_in)
    bnorm = BatchNormalization()(conv)
    x_out = activations.gelu(bnorm)
    return x_out

def channel_interaction(x_in):
    conv1 = Conv2D(x_in.shape[-1] // 8, (1,1))(x_in)
    bnorm = BatchNormalization()(conv1)
    gelu = activations.gelu(bnorm)
    conv2 = Conv2D(x_in.shape[-1] // 2, (1,1))(gelu)

    return conv2

def spatial_interaction(x_in):
    conv1 = Conv2D(x_in.shape[-1] // 8, (1,1))(x_in)
    bnorm = BatchNormalization()(conv1)
    gelu = activations.gelu(bnorm)
    conv2 = Conv2D(1, (1,1))(gelu)
    return conv2

def MIX(x_in, bias):

    num_heads = 8
    windows = window_partition(x_in)
    # ?
    proj_attn = Dense(windows.shape[-1]//2)(windows)
    proj_cnn = Dense(windows.shape[-1])(windows)
    lnorm1 = LayerNormalization()(proj_attn)
    lnorm2 = LayerNormalization()(proj_cnn)
    #

    x_cnn = window_reverse2(lnorm2, H=x_in.shape[1], W=x_in.shape[2], C=proj_cnn.shape[-1])
    dwconv3x3 = dw_conv_block(transpose(x_cnn, perm=[0, 2, 3, 1]))
    c_interact = channel_interaction(GlobalAveragePooling2D(keepdims=True)(dwconv3x3))
    c_interact = Conv2D(windows.shape[-1] // 2, (1,1))(c_interact)

    expanded = Dense(lnorm1.shape[-1]*3)(lnorm1)
    
    B, N, C = lnorm1.shape
    qkv = transpose(reshape(expanded,(B, N, 3, num_heads, C//num_heads)),perm=[2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    x_cnn2v = activations.sigmoid(c_interact)
    x_cnn2v = reshape(transpose(x_cnn2v, perm=[0,3,1,2]), (-1, 1, num_heads, 1, C//num_heads))
    v = reshape(v, (x_cnn2v.shape[0], -1, num_heads, N, C // num_heads))
    v = v*x_cnn2v
    v = reshape(v, (-1, num_heads, N, C//num_heads))

    attn_dim = x_in.shape[-1] // 2
    head_dim = attn_dim // num_heads
    scale = head_dim ** -0.5

    q = q*scale
    k = transpose(k, [0, 1, 3, 2])
    attn = (q @ k)
    attn = attn+bias
    attn = Softmax()(attn)
    # attn = Dropout()(attn)

    x_atten = reshape(transpose((attn @ v), (0, 2, 1, 3)), (B, N, C))
    x_spatial = transpose(window_reverse2(x_atten, H=x_in.shape[1], W=x_in.shape[2], C=C), perm=[0,2,3,1])
    s_interact = spatial_interaction(x_spatial)
    x_cnn = activations.sigmoid(s_interact) * transpose(x_cnn, perm=[0,2,3,1])
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = window_partition2(transpose(x_cnn, perm=[0,3,1,2]))

    x_atten = LayerNormalization()(x_atten)
    x = concat([x_atten, x_cnn], -1)
    x = Dense(x_in.shape[-1])(x)

    return x
##############################################

def MLP(x_in):
    dense1 = Dense(512*4)(x_in)
    relu1 = ReLU()(dense1)
    drop1 = Dropout(0.)(relu1)
    dense2 = Dense(512)(drop1)
    drop2 = Dropout(0.)(dense2)

    return drop2

def mix_former(x_in, bias):
    lnorm1 = LayerNormalization()(x_in)
    mix = MIX(lnorm1, bias)
    mix = reshape(mix, (-1, 8, 8, x_in.shape[-1]))
    mix = window_reverse(mix, 8, x_in.shape[1], x_in.shape[2], x_in.shape[3])
    cat1 = x_in + mix

    lnorm2 = LayerNormalization()(cat1)
    mlp = MLP(lnorm2)
    x_out = cat1 + mlp
    return x_out

def down_conv_block(x_in, filters):
    maxp = MaxPooling2D((2,2))(x_in)
    x_out = conv_block(maxp, filters=filters)
    x_out = conv_block(x_out, filters=filters)
    # skip = UpSampling2D(size=(2,2))(x_out)
    return x_out

def up_conv_block(x_in, skip, filters):
    ups = UpSampling2D(size=(2,2))(x_in)
    conv = conv_block(ups, filters)
    x_out = concat([conv, skip], -1)
    return x_out

def AMM(x_in, dim):
    
    s_conv1 = DepthwiseConv2D((3,3), padding='same')(x_in)
    s_conv2 = Conv2D(dim//16, (1,1))(s_conv1)
    s_bnorm = BatchNormalization()(s_conv2)
    s_relu = ReLU()(s_bnorm)
    s_conv3 = Conv2D(1, (1,1), activation='sigmoid')(s_relu)

    c_pool = GlobalAveragePooling2D(keepdims=True)(x_in)
    c_conv1 = Conv2D(dim//16, (1,1))(c_pool)
    c_relu = ReLU()(c_conv1)
    c_conv2 = Conv2D(dim, (1,1), activation='sigmoid')(c_relu)

    spatial_branch = s_conv3 * x_in
    channel_branch = c_conv2 * x_in

    x_out = spatial_branch + channel_branch
    return x_out

def down_mix_former(x_in):

    # return x_out, skip
    pass

def up_mix_former(x_in, skip, bias1, bias2):

    up = UpSampling2D(size=(2,2))(x_in)
    cat = concat([up, skip], -1)
    conv1 = conv_block(cat, 512)
    mixf = mix_former(conv1,bias1)
    x_out = mix_former(mixf, bias2)

    return x_out
# [64, 128, 256, 512]
# base_c = 64
# factor = 2
# num_heads = 8

def get_model(input_size, bias_size, batch_size=16, format='channels_last'):

    nn_in = Input(shape=input_size, batch_size=batch_size)
    biases = Input(shape=bias_size[1:], batch_size=bias_size[0])
    bias1, bias2, bias3, bias4, bias5, bias6 = unstack(biases)
    down_conv1a = conv_block(nn_in, 64)
    down_conv1b = conv_block(down_conv1a, 64)
    down_conv2 = down_conv_block(down_conv1b, 128)
    down_conv3 = down_conv_block(down_conv2, 256)

    maxp1 = MaxPooling2D((2,2))(down_conv3)
    down_conv4 = conv_block(maxp1, 512)
    down_mformer1a = mix_former(down_conv4, bias1)
    down_mformer1b = mix_former(down_mformer1a, bias2)

    maxp2 = MaxPooling2D((2,2))(down_mformer1b)
    down_conv5 = conv_block(maxp2, 512)
    down_mformer2a = mix_former(down_conv5, bias3)
    down_mformer2b = mix_former(down_mformer2a, bias4)

    amm1 = AMM(down_conv2, 128)
    amm2 = AMM(down_conv3, 256)
    amm3 = AMM(down_mformer1b, 512)

    up_mix = up_mix_former(down_mformer2b, amm3, bias5, bias6)

    up_conv1 = up_conv_block(up_mix, amm2, 256)
    up_conv2 = up_conv_block(up_conv1, amm1, 128)
    up_conv3 = up_conv_block(up_conv2, down_conv1b, 64)

    out = Conv2D(2, (1,1), padding='valid', data_format=format)(up_conv3)
    out = Softmax(axis=-1)(out)

    model = Model(inputs=[nn_in, biases], outputs=out, name="MU_Net")

    return model

if __name__ == '__main__':
    t = random.uniform(shape=[512, 512, 512])
    # t = random.uniform(shape=[16, 512, 512, 512])
    b1 = get_bias()
    b2 = get_bias()
    b3 = get_bias()
    b4 = get_bias()
    b5 = get_bias()
    b6 = get_bias()
    # m = MIX(t, b)
    # AMM(t)
    biases = stack([b1,b2,b3,b4,b5,b6])
    m = get_model(t.shape, biases.shape)

    print('a')