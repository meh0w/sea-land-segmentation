from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D, ReLU, Softmax
from tensorflow.keras import Sequential, optimizers, Model
from tensorflow import name_scope, concat

def bn_relu(inputs):
    x = BatchNormalization()(inputs)
    return ReLU()(x)

def conv_bn_relu(inputs, kenrel=(3, 3), stride=(1, 1), pad='valid', num_filter=None, name=None, format='channels_last'):
    return bn_relu(Conv2D(num_filter, kenrel, strides=stride, padding=pad, data_format=format)(inputs))

def down_block(inputs, format='channels_last'):
    # x = MaxPooling2D((2, 2), strides=(2,2), data_format=format)(inputs)
    # temp = conv_bn_relu(x, (3, 3), (1, 1), 'same', 2*num_filters)
    # temp = Conv2D(num_filters, 3, strides=(1, 1), data_format=format, padding='same')(temp)
    # # bn = BatchNormalization()(temp)
    # temp += x
    # act = bn_relu(temp)
    x = bn_relu(inputs)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    x = bn_relu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format=format)(x)

    plus_out = x + inputs

    out = MaxPooling2D((2, 2), strides=(2,2), data_format=format)(inputs)

    return plus_out, out

def up_block(inputs, plus_in, format='channels_last'):

    up_sampled = UpSampling2D(size=(2,2), data_format=format)(inputs)
    x = bn_relu(up_sampled)
    x = concat([x, plus_in], -1)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    x = bn_relu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format=format)(x)

    out = up_sampled + x

    # x = UpSampling2D(size=(2,2), data_format=format)(act)
    # temp = concatenate([bn, x], axis=-1)
    # temp = conv_bn_relu(temp, (3, 3), (1, 1), 'same', 2*num_filters)
    # conv = Conv2D(num_filters, (3, 3), strides=(1, 1), padding='same', data_format=format)(temp)
    # bn = BatchNormalization()(conv)
    # bn += x
    # act = ReLU()(bn)

    return out

def get_model(input_size, batch_size, format='channels_last'):

    nn_in = Input(shape=input_size)

    ######## DOWN 1 ##########
    x = Conv2D(32, 3, padding='same', data_format=format)(nn_in)
    down_plus1, down1 = down_block(x)

    down_plus2, down2 = down_block(down1)

    down_plus3, down3 = down_block(down2)

    down_plus4, down4 = down_block(down3)

    down_plus5, down5 = down_block(down4)

    down_plus6, down6 = down_block(down5)

    down_plus7, down7 = down_block(down6)

    up7 = up_block(down7, down_plus7)

    up6 = up_block(up7, down_plus6)

    up5 = up_block(up6, down_plus5)

    up4 = up_block(up5, down_plus4)

    up3 = up_block(up4, down_plus3)

    up2 = up_block(up3, down_plus2)

    up1 = up_block(up2, down_plus1)

    relu = bn_relu(up1)
    out = Conv2D(2, (1,1), padding='valid', data_format=format)(relu)
    out = Softmax(axis=-1)(out)
    # net = conv_bn_relu(x, (3, 3), (1, 1), 'same', 64)
    # conv_bn1 = Conv2D(32, 3, padding='same', data_format=format)(net)
    # bn1 = BatchNormalization()(conv_bn1)
    # act1 = ReLU()(bn1)

    # bn2, act2 = down_block(act1, 32)
    # bn3, act3 = down_block(act2, 32)
    # bn4, act4 = down_block(act3, 32)
    # bn5, act5 = down_block(act4, 32)
    # bn6, act6 = down_block(act5, 32)
    # bn7, act7 = down_block(act6, 32)

    # temp = up_block(act7, bn6, 32)
    # temp = up_block(temp, bn5, 32)
    # temp = up_block(temp, bn4, 32)
    # score4 = Conv2D(2, (1,1), padding='valid', data_format=format)(temp)
    # net4 = Softmax()(score4)

    # temp = up_block(temp, bn3, 32)
    # score3 = Conv2D(2, (1,1), padding='valid', data_format=format)(temp)
    # net3 = Softmax()(score3)

    # temp = up_block(temp, bn2, 32)
    # score2 = Conv2D(2, (1,1), padding='valid', data_format=format)(temp)
    # net2 = Softmax()(score2)

    # temp = up_block(temp, bn1, 32)
    # score1 = Conv2D(2, (1,1), padding='valid', data_format=format)(temp)
    # net1 = Softmax(axis=-1)(score1)


    # model = Model(inputs=nn_in, outputs=[net1, net2, net3, net4], name="DeepUNet")
    model = Model(inputs=nn_in, outputs=out, name="DeepUNet")

    return model

