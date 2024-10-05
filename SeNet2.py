from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D, ReLU, Softmax, Conv2DTranspose
from tensorflow.keras import Sequential, optimizers, Model
from tensorflow import name_scope, nn
import tensorflow as tf
import numpy as np

"""
    Implementation of SeNet described in article

    Some details are not described such as:
    - size of convolution kernel


    Requires data to be in shape of (n, height, width, channels)
"""

def unpool(pool, ind, ksize=[1, 2, 2, 1], output_shape=None, b_size=1, scope='unpool'):
    return UpSampling2D(size=(2,2), interpolation='nearest')(pool)

def bn_relu(inputs):
    x = BatchNormalization()(inputs)
    return ReLU()(x)

def conv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None):
    return bn_relu(Conv2D(num_filter, kenrel, strides=stride, padding=pad, data_format='channels_last')(inputs))

def deconv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None):
    return bn_relu(Conv2DTranspose(num_filter, kenrel, strides=stride, padding=pad, data_format='channels_last')(inputs))

class max_pool_with_argmax(Layer):
    def call(self, x):
        return nn.max_pool_with_argmax(x, (2, 2), strides=(2,2), padding='SAME', data_format='NHWC')

def get_model(input_size, batch_size):

    nn_in = Input(shape=input_size)
    cbnr_1 = conv_bn_relu(nn_in, (3, 3), (1, 1), 'same', 32)
    cbnr_2 = conv_bn_relu(cbnr_1, (3, 3), (1, 1), 'same', 32)

    maxp_1, amax_1 = max_pool_with_argmax()(cbnr_2)

    cbnr_3 = conv_bn_relu(maxp_1, (3, 3), (1, 1), 'same', 64)
    cbnr_4 = conv_bn_relu(cbnr_3, (3, 3), (1, 1), 'same', 64)

    maxp_2, amax_2 = max_pool_with_argmax()(cbnr_4)

    cbnr_5 = conv_bn_relu(maxp_2, (3, 3), (1, 1), 'same', 128)
    cbnr_6 = conv_bn_relu(cbnr_5, (3, 3), (1, 1), 'same', 128)
    cbnr_7 = conv_bn_relu(cbnr_6, (3, 3), (1, 1), 'same', 128)

    maxp_3, amax_3 = max_pool_with_argmax()(cbnr_7)

    shape = np.array(cbnr_7.shape)
    shape[0] = batch_size
    umaxp_1 = unpool(maxp_3, amax_3, [1,2,2,1], shape, batch_size)

    dcbnr_1 = deconv_bn_relu(umaxp_1, (3, 3), (1, 1), 'same', 128)
    dcbnr_2 = deconv_bn_relu(dcbnr_1, (3, 3), (1, 1), 'same', 128)
    dcbnr_3 = deconv_bn_relu(dcbnr_2, (3, 3), (1, 1), 'same', 128)

    shape = np.array(cbnr_4.shape)
    shape[0] = batch_size
    umaxp_2 = unpool(dcbnr_3, amax_2, [1,2,2,1], shape, batch_size)

    dcbnr_4 = deconv_bn_relu(umaxp_2, (3, 3), (1, 1), 'same', 64)
    dcbnr_5 = deconv_bn_relu(dcbnr_4, (3, 3), (1, 1), 'same', 64)

    shape = np.array(cbnr_2.shape)
    shape[0] = batch_size
    umaxp_2 = unpool(dcbnr_5, amax_1, [1,2,2,1], shape, batch_size)

    dcbnr_6 = conv_bn_relu(umaxp_2, (3, 3), (1, 1), 'same', 32)
    dcbnr_7 = conv_bn_relu(dcbnr_6, (3, 3), (1, 1), 'same', 32)

    # branching out into segmentation layer and edge detection layer
    con_seg = Conv2D(2, (3,3), strides=(1,1), padding='same', data_format='channels_last')(dcbnr_7)

    smax_seg = Softmax(axis=-1, name='output_seg')(con_seg)

    con_edge = Conv2D(2, (3,3), strides=(1,1), padding='same', data_format='channels_last')(dcbnr_7)

    smax_edge = Softmax(axis=-1, name='output_edge')(con_edge)

    model = Model(inputs=nn_in, outputs=[smax_seg, smax_edge], name="SeNet2")

    return model
