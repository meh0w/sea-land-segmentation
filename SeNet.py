from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D, ReLU, Softmax, Conv2DTranspose
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
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.name_scope(scope):
        input_shape = tf.shape(pool)

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = output_shape
        ret.set_shape(set_output_shape)
        return ret

def bn_relu(inputs):
    x = BatchNormalization()(inputs)
    return ReLU()(x)

def conv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None):
    return bn_relu(Conv2D(num_filter, kenrel, strides=stride, padding=pad, data_format='channels_last')(inputs))

def deconv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None):
    return bn_relu(Conv2DTranspose(num_filter, kenrel, strides=stride, padding=pad, data_format='channels_last')(inputs))


def get_model(input_size, batch_size):

    nn_in = Input(shape=input_size)
    cbnr_1 = conv_bn_relu(nn_in, (3, 3), (1, 1), 'same', 32)
    cbnr_2 = conv_bn_relu(cbnr_1, (3, 3), (1, 1), 'same', 32)

    maxp_1, amax_1 = nn.max_pool_with_argmax(cbnr_2, (2, 2), strides=(2,2), padding='SAME', data_format='NHWC')

    cbnr_3 = conv_bn_relu(maxp_1, (3, 3), (1, 1), 'same', 64)
    cbnr_4 = conv_bn_relu(cbnr_3, (3, 3), (1, 1), 'same', 64)

    maxp_2, amax_2 = nn.max_pool_with_argmax(cbnr_4, (2, 2), strides=(2,2), padding='SAME', data_format='NHWC')

    cbnr_5 = conv_bn_relu(maxp_2, (3, 3), (1, 1), 'same', 128)
    cbnr_6 = conv_bn_relu(cbnr_5, (3, 3), (1, 1), 'same', 128)
    cbnr_7 = conv_bn_relu(cbnr_6, (3, 3), (1, 1), 'same', 128)

    maxp_3, amax_3 = nn.max_pool_with_argmax(cbnr_7, (2, 2), strides=(2,2), padding='SAME', data_format='NHWC')

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

    con = Conv2D(2, (3,3), strides=(1,1), padding='same', data_format='channels_last')(dcbnr_7)

    smax = Softmax()(con)

    model = Model(inputs=nn_in, outputs=smax, name="model")

    return model
