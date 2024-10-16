import numpy as np
import os
from tifffile import tifffile
import matplotlib.pyplot as plt
import pickle
# import tensorflow as tf

# a utility function to add weight decay after the model is defined.
def add_weight_decay(model, weight_decay):
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with tf.keras.backend.name_scope('weight_regularizer'):
                    regularizer = lambda param=param: tf.keras.regularizers.l2(factor)(param)
                    m.add_loss(regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay/2.0)
    return

def to_sparse(im):
    sparse_im = np.zeros((2,)+im.shape)
    idx = np.argwhere(im == 1)
    sparse_im[1, idx[:,0], idx[:,1]] = 1
    idx = np.argwhere(im == 0)
    sparse_im[0, idx[:,0], idx[:,1]] = 1

    return sparse_im

def flatten(im):
    return np.argmax(im, 0)

def load_data(path, part):
    ims = []
    labs = []

    if part == 'test':
        for file in os.listdir(rf'{path}/images'):
            if file.endswith('.tif'):
                ims.append(tifffile.imread(rf'{path}/images/{file}')[:,:,:])

        for file in os.listdir(rf'{path}/labels'):
            if file.endswith('.tif'):
                labs.append(to_sparse(tifffile.imread(rf'{path}/labels/{file}')))

    elif part == 'train':
        for file in os.listdir(rf'{path}/images'):
            if file.endswith('.npy'):
                ims.append(np.load(rf'{path}/images/{file}')[:,:,:]/24_000)

        for file in os.listdir(rf'{path}/labels'):
            if file.endswith('.npy'):
                labs.append(to_sparse(np.load(rf'{path}/labels/{file}')[0]))

    ims = np.moveaxis(np.asarray(ims), 1, -1)
    labs = np.asarray(labs)

    return ims, labs

def get_file_names(path, extension, dataset="SWED"):

    if dataset == "SWED" or dataset == "SWED_FULL":
        # exclude image+label pairs with incorrect labels
        # (values different than 0 or 1)
        with open('utils/exclude_imgs.pickle', 'rb') as f:
            excluded_imgs = pickle.load(f) 

        with open('utils/exclude_labels.pickle', 'rb') as f:
            excluded_labels = pickle.load(f)
            
        ims, labs = [], []
        for file in os.listdir(rf'{path}/images'):
            if file.endswith(extension):
                if file not in excluded_imgs:
                    ims.append(rf'{path}/images/{file}')

        for file in os.listdir(rf'{path}/labels'):
            if file.endswith(extension):
                if file not in excluded_labels:
                    labs.append(rf'{path}/labels/{file}')
        return np.asarray(ims), np.asarray(labs)
    elif dataset == "SNOWED":
        # due to file structure we only need folder names
        folders = os.listdir(path)

        return np.asarray(folders)

def shuffle(im, lab):
    p = np.random.permutation(len(im))

    return im[p], lab[p]

def compare_imgs(im1, im2):
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(im1)
    ax[1].imshow(im2)

    plt.show()
    
