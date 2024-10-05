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


class Sorensen_Dice(tf.keras.losses.Loss):
  @tf.function
  def call(self, y_true, y_pred):
        epsilon = 1e-9
        return tf.reduce_mean(1 - ((2*tf.reduce_sum(y_true*y_pred, axis=(1,2,3))+epsilon)/(tf.reduce_sum(y_true+y_pred, axis=(1,2,3))+epsilon)))


def sob(image):
  sobel_x = tf.constant([[[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]], [[[-2, -2], [-2, -2]], [[0, 0], [0, 0]], [[2, 2], [2, 2]]], [[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]]], tf.float32)
  sobel_x_filter = tf.reshape(sobel_x, [3, 3, 2, 2])
  sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

  filtered_x = tf.nn.conv2d(image, sobel_x_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
  filtered_y = tf.nn.conv2d(image, sobel_y_filter,
                            strides=[1, 1, 1, 1], padding='SAME')
  


class Sobel_loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        Gx_kernel = tf.constant([[[[-1]], [[0]], [[1]]], [[[-2]], [[0]], [[2]]], [[[-1]], [[0]], [[1]]]], tf.float32)
        # Gx_kernel = tf.constant([[[[-1], [-1]], [[0], [0]], [[1], [1]]], [[[-2], [-2]], [[0], [0]], [[2], [2]]], [[[-1], [-1]], [[0], [0]], [[1], [1]]]], tf.float32)
        # Gx_kernel = tf.constant([[[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]], [[[-2, -2], [-2, -2]], [[0, 0], [0, 0]], [[2, 2], [2, 2]]], [[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]]], tf.float32)
        Gy_kernel = tf.transpose(Gx_kernel, [1, 0, 2, 3])

        y_pred_ = tf.expand_dims(y_pred[:,:,:,0], -1)
        y_true_ = tf.expand_dims(y_true[:,:,:,0], -1)
        Gx_pred = tf.nn.conv2d(y_pred_, Gx_kernel, strides=[1, 1, 1, 1], padding='SAME')
        Gy_pred = tf.nn.conv2d(y_pred_, Gy_kernel, strides=[1, 1, 1, 1], padding='SAME')

        Gx_true = tf.nn.conv2d(y_true_, Gx_kernel, strides=[1, 1, 1, 1], padding='SAME')
        Gy_true = tf.nn.conv2d(y_true_, Gy_kernel, strides=[1, 1, 1, 1], padding='SAME')

        G_pred = tf.math.sqrt(tf.math.square(Gx_pred) + tf.math.square(Gy_pred))
        G_true = tf.math.sqrt(tf.math.square(Gx_true) + tf.math.square(Gy_true))

        loss = tf.reduce_mean(tf.math.square(G_true-G_pred))
        return loss


class Weighted_Dice(tf.keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        epsilon = 1e-9
        Gx_kernel = tf.constant([[[[-1]], [[0]], [[1]]], [[[-2]], [[0]], [[2]]], [[[-1]], [[0]], [[1]]]], tf.float32)
        # Gx_kernel = tf.constant([[[[-1], [-1]], [[0], [0]], [[1], [1]]], [[[-2], [-2]], [[0], [0]], [[2], [2]]], [[[-1], [-1]], [[0], [0]], [[1], [1]]]], tf.float32)
        # Gx_kernel = tf.constant([[[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]], [[[-2, -2], [-2, -2]], [[0, 0], [0, 0]], [[2, 2], [2, 2]]], [[[-1, -1], [-1, -1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]]], tf.float32)
        Gy_kernel = tf.transpose(Gx_kernel, [1, 0, 2, 3])

        y_pred = tf.expand_dims(y_pred[:,:,:,0], -1)
        y_true = tf.expand_dims(y_true[:,:,:,0], -1)

        Gx_true = tf.nn.conv2d(y_true, Gx_kernel, strides=[1, 1, 1, 1], padding='SAME')
        Gy_true = tf.nn.conv2d(y_true, Gy_kernel, strides=[1, 1, 1, 1], padding='SAME')

        W = tf.math.sqrt(Gx_true**2 + Gy_true**2)
        return tf.reduce_mean(1 - ((2*tf.reduce_sum(y_true*W*y_pred, axis=(1,2,3))+epsilon)/(tf.reduce_sum(y_true+W*y_pred, axis=(1,2,3))+epsilon)))
  
class CEDice(tf.keras.losses.Loss):
    def __init__(self):
       super().__init__()
       self.ce = tf.keras.losses.CategoricalCrossentropy()
    @tf.function
    def call(self, y_true, y_pred):
      epsilon = 1e-9
      dice=tf.reduce_mean(1 - ((2*tf.reduce_sum(y_true*y_pred, axis=(1,2,3))+epsilon)/(tf.reduce_sum(y_true+y_pred, axis=(1,2,3))+epsilon)))
      ce = self.ce(y_true, y_pred)

      return ce + 0.7*dice