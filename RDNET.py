from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, PReLU, Input, concatenate, UpSampling2D
from tensorflow.keras import Sequential, optimizers, Model
from tensorflow import name_scope


def dense_block(inputs, block_name='unnamed_dense_block', filters=64):
    with name_scope(block_name):
        concatenated_inputs = inputs

        conv1 = Conv2D(filters, 1, padding='same', data_format='channels_first')(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, conv1], axis=1)
        conv2 = Conv2D(filters, 3, padding='same', data_format='channels_first')(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, conv2], axis=1)
        conv3 = Conv2D(filters, 3, padding='same', data_format='channels_first')(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, conv3], axis=1)
        conv4 = Conv2D(filters, 3, padding='same', data_format='channels_first')(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, conv4], axis=1)
        conv5 = Conv2D(filters, 3, padding='same', data_format='channels_first')(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, conv5], axis=1)
        conv6 = Conv2D(filters, 1, padding='same', data_format='channels_first')(concatenated_inputs)
        # concatenated_inputs = concatenate([concatenated_inputs, conv6], axis=1)
        bn = BatchNormalization()(conv6)
        out = PReLU()(bn+inputs)

    return out

def get_model(input_size):
    ######## DOWN 1 ##########
    nn_in = Input(shape=input_size)
    x = Conv2D(64, 3, data_format='channels_first', padding='same')(nn_in)
    x = dense_block(x, 'dense1', 64)

    # BN, PReLU, Conv - most likely reverse order (possibly padding=same)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    d_1 = Conv2D(64, 2, data_format='channels_first', padding='same')(x)

    x = Conv2D(64, 2, strides=(2,2), data_format='channels_first')(d_1)

    ######## DOWN 2 ##########
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(128, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense2', 128)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    d_2 = Conv2D(128, 2, data_format='channels_first', padding='same')(x)
    x = Conv2D(128, 2, strides=(2,2), data_format='channels_first')(d_2)

    ######## DOWN 3 ##########
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(256, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense3', 256)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    d_3 = Conv2D(256, 2, data_format='channels_first', padding='same')(x)
    x = Conv2D(256, 2, strides=(2,2), data_format='channels_first')(d_3)

    ######## DOWN 4 ##########
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(512, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense4', 512)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    d_4 = Conv2D(512, 2, data_format='channels_first', padding='same')(x)
    x = Conv2D(512, 2, strides=(2,2), data_format='channels_first')(d_4)

    ######## BRIDGE ##########

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(1024, 2, data_format='channels_first', padding='same')(x)

    x = dense_block(x, 'dense5', 1024)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(1024, 2, data_format='channels_first', padding='same')(x)

    ######## UP4 ##########
    x = UpSampling2D(size=(2,2), data_format='channels_first')(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(512, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense4', 512)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(512, 2, data_format='channels_first', padding='same')(x)
    x = x + d_4

    ######## UP3 ##########
    x = UpSampling2D(size=(2,2), data_format='channels_first')(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(256, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense4', 256)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(256, 2, data_format='channels_first', padding='same')(x)

    x = x + d_3
    ######## UP2 ##########
    x = UpSampling2D(size=(2,2), data_format='channels_first')(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(128, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense4', 128)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(128, 2, data_format='channels_first', padding='same')(x)

    x = x + d_2
    ######## UP1 ##########
    x = UpSampling2D(size=(2,2), data_format='channels_first')(x)

    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(64, 3, data_format='channels_first', padding='same')(x)
    
    x = dense_block(x, 'dense4', 64)
    
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(64, 2, data_format='channels_first', padding='same')(x)
    x = x + d_1
    
    x = Conv2D(1, 1, data_format='channels_first', padding='same')(x)


    model = Model(inputs=nn_in, outputs=x, name="model")
    return model