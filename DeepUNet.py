from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D, ReLU, Softmax
from tensorflow.keras import Sequential, optimizers, Model
from tensorflow import name_scope

def bn_relu(inputs):
    x = BatchNormalization()(inputs)
    return ReLU()(x)

def conv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None):
    return bn_relu(Conv2D(num_filter, kenrel, strides=stride, padding=pad, data_format='channels_first')(inputs))

def down_block(inputs, num_filters):
    x = MaxPooling2D((2, 2), strides=(2,2), data_format='channels_first')(inputs)
    temp = conv_bn_relu(x, (3, 3), (1, 1), 'same', 2*num_filters)
    temp = Conv2D(num_filters, 3, strides=(1, 1), data_format='channels_first', padding='same')(temp)
    bn = BatchNormalization()(temp)
    bn += x
    act = ReLU()(bn)

    return bn, act

def up_block(act, bn, num_filters):
    x = UpSampling2D(size=(2,2), data_format='channels_first')(act)
    temp = concatenate([bn, x], axis=1)
    temp = conv_bn_relu(temp, (3, 3), (1, 1), 'same', 2*num_filters)
    conv = Conv2D(num_filters, (3, 3), strides=(1, 1), padding='same', data_format='channels_first')(temp)
    bn = BatchNormalization()(conv)
    bn += x
    act = ReLU()(bn)

    return act

def get_model(input_size):

    nn_in = Input(shape=input_size)

    ######## DOWN 1 ##########
    x = conv_bn_relu(nn_in, (3, 3), (1, 1), 'same', 32)
    net = conv_bn_relu(x, (3, 3), (1, 1), 'same', 64)
    conv_bn1 = Conv2D(32, 3, padding='same', data_format='channels_first')(net)
    bn1 = BatchNormalization()(conv_bn1)
    act1 = ReLU()(bn1)

    bn2, act2 = down_block(act1, 32)
    bn3, act3 = down_block(act2, 32)
    bn4, act4 = down_block(act3, 32)
    bn5, act5 = down_block(act4, 32)
    bn6, act6 = down_block(act5, 32)
    bn7, act7 = down_block(act6, 32)

    temp = up_block(act7, bn6, 32)
    temp = up_block(temp, bn5, 32)
    temp = up_block(temp, bn4, 32)
    score4 = Conv2D(2, (1,1), padding='valid', data_format='channels_first')(temp)
    net4 = Softmax()(score4)

    temp = up_block(temp, bn3, 32)
    score3 = Conv2D(2, (1,1), padding='valid', data_format='channels_first')(temp)
    net3 = Softmax()(score3)

    temp = up_block(temp, bn2, 32)
    score2 = Conv2D(2, (1,1), padding='valid', data_format='channels_first')(temp)
    net2 = Softmax()(score2)

    temp = up_block(temp, bn1, 32)
    score1 = Conv2D(2, (1,1), padding='valid', data_format='channels_first')(temp)
    net1 = Softmax(axis=2)(score1)


    # model = Model(inputs=nn_in, outputs=[net1, net2, net3, net4], name="DeepUNet")
    model = Model(inputs=nn_in, outputs=net1, name="DeepUNet")

    return model

