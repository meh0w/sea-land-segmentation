import tensorflow as tf
import numpy as np



class SeNet_loss(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    reg_term = 30
    y_pred_neighbours = tf.image.extract_patches(y_pred, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    y_pred_neighbours = tf.reshape(y_pred_neighbours, (tf.shape(y_pred_neighbours)[0], y_pred_neighbours.shape[1], y_pred_neighbours.shape[2], 9, -1))
    inputs = y_true[:,:,:,2:]
    y_true = y_true[:,:,:,:2]

    lab_shape = tf.concat([tf.shape(y_pred)[:-1], [1]], axis=0)

    lab_0 = tf.tile(tf.reshape(y_pred[:, :, :, 0], lab_shape), [1, 1, 1, 9])
    lab_0 = lab_0[:,1:-1,1:-1,:]  #p_0,i

    lab_1 = tf.tile(tf.reshape(y_pred[:, :, :, 0], lab_shape), [1, 1, 1, 9])
    lab_1 = lab_1[:,1:-1,1:-1,:]  #p_0,i

    img = tf.image.extract_patches(inputs, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    img = tf.reshape(img, (tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], 9, -1))

    img_shape = tf.concat([tf.shape(inputs)[:-1], [1], [tf.shape(inputs)[-1]]], 0)
    img_tiled = tf.tile(tf.reshape(inputs, img_shape), [1, 1, 1, 9, 1])
    img_tiled = img_tiled[:,1:-1,1:-1,:,:]

    variance_term = tf.reduce_mean((img_tiled - img)**2, axis = (1,2,3,4))
    variance_shape = tf.concat([tf.shape(variance_term), [1,1,1]], axis=0)
    variance_term = tf.tile(tf.reshape(variance_term, variance_shape), [1, 254, 254, 9])
    c = tf.reduce_sum((img_tiled - img)**2, axis=4)
    
    p_li = tf.where(tf.argmax(y_true, axis=3) == 1, y_pred[:,:,:,1], y_pred[:,:,:,0])
    p_li = p_li[:,1:-1,1:-1]

    loss = tf.reduce_mean(-tf.math.log(p_li) + (reg_term/2)*tf.reduce_sum(((lab_0 - y_pred_neighbours[:,:,:,:,0])**2 + (lab_0 - y_pred_neighbours[:,:,:,:,1])**2)*tf.cast(tf.exp(-c/variance_term), tf.float32), axis=-1))


    return loss
  
def dummy(y_true, y_pred):
    reg_term = 1
    y_pred_neighbours = tf.image.extract_patches(y_pred, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    y_pred_neighbours = tf.reshape(y_pred_neighbours, (tf.shape(y_pred_neighbours)[0], y_pred_neighbours.shape[1], y_pred_neighbours.shape[2], 9, -1))
    inputs = y_true[:,:,:,2:]
    y_true = y_true[:,:,:,:2]

    lab_shape = tf.concat([tf.shape(y_pred)[:-1], [1]], axis=0)

    lab_0 = tf.tile(tf.reshape(y_pred[:, :, :, 0], lab_shape), [1, 1, 1, 9])
    lab_0 = lab_0[:,1:-1,1:-1,:]  #p_0,i
    # lab_0 = tf.constant(lab_0)

    lab_1 = tf.tile(tf.reshape(y_pred[:, :, :, 0], lab_shape), [1, 1, 1, 9])
    lab_1 = lab_1[:,1:-1,1:-1,:]  #p_0,i

    # (p_0,i - p_0,j)^2 + (p_1,i - p_1,j)^2

    img = tf.image.extract_patches(inputs, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    img = tf.reshape(img, (tf.shape(img)[0], img.shape[1], img.shape[2], 9, -1))

    img_shape = tf.concat([inputs.shape[:-1], [1], [inputs.shape[-1]]], 0)
    img_tiled = tf.tile(tf.reshape(inputs, img_shape), [1, 1, 1, 9, 1])
    img_tiled = img_tiled[:,1:-1,1:-1,:,:]

    variance_term = tf.reduce_mean((img_tiled - img)**2, axis = (1,2,3,4))
    variance_shape = tf.concat([tf.shape(variance_term), [1,1,1]], axis=0)
    variance_term = tf.tile(tf.reshape(variance_term, variance_shape), [1, 254, 254, 9])
    c = tf.reduce_sum((img_tiled - img)**2, axis=4)
    
    p_li = tf.where(tf.argmax(y_true, axis=3) == 1, y_pred[:,:,:,1], y_pred[:,:,:,0])
    p_li = p_li[:,1:-1,1:-1]

    loss = tf.reduce_mean(-tf.math.log(p_li) + (reg_term/2)*tf.reduce_sum(((lab_0 - y_pred_neighbours[:,:,:,:,0])**2 + (lab_0 - y_pred_neighbours[:,:,:,:,1])**2)*tf.cast(tf.exp(-c/variance_term), tf.float32), axis=-1))


    return loss

if __name__ == '__main__':
    img = np.arange(256*256*14).reshape(1, 256, 256, 14).astype(np.float)
    lab = np.arange(256*256*2).reshape(1, 256, 256, 2).astype(np.float)

    dummy(img, lab)

    out_img = tf.image.extract_patches(img, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    out_lab = tf.image.extract_patches(lab, [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    out_reshaped = tf.reshape(out_img, (tf.shape(out_img)[0], out_img.shape[1], out_img.shape[2], 9, -1)) # [batch_size, height, width, n, channels]
    lab_reshaped = tf.reshape(out_lab, (out_lab.shape[0], out_lab.shape[1], out_lab.shape[2], 9, -1)) # [batch_size, height, width, n, channels]    p_li,j

    print('a')
    lab_shape = tf.concat([tf.shape(lab)[:-1], [1]], axis=0)

    lab_0 = tf.tile(tf.reshape(lab[:, :, :, 0], lab_shape), [1, 1, 1, 9])
    lab_0 = lab_0[:,1:-1,1:-1,:]  #p_0,i

    lab_1 = tf.tile(tf.reshape(lab[:, :, :, 1], lab_shape), [1, 1, 1, 9])
    lab_1 = lab_1[:,1:-1,1:-1,:]  #p_1,i

    img_shape = tf.concat([img.shape[:-1], [1], [img.shape[-1]]], 0)
    img_tiled = tf.tile(tf.reshape(img, img_shape), [1, 1, 1, 9, 1])
    img_tiled = img_tiled[:,1:-1,1:-1,:,:]

    c = (img_tiled - out_reshaped)**2
    variance_term = tf.reduce_mean(c, axis = (1,2,3,4))
# (p_0,i - p_0,j)^2 + (p_1,i - p_1,j)^2