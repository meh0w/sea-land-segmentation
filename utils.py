import numpy as np
import os
from tifffile import tifffile

def to_sparse(im):
    sparse_im = np.zeros((2,)+im.shape)
    idx = np.argwhere(im == 1)
    sparse_im[1, idx[:,0], idx[:,1]] = 1

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

def get_file_names(path):

    ims, labs = [], []
    for file in os.listdir(rf'{path}/images'):
        if file.endswith('.npy'):
            ims.append(rf'{path}/images/{file}')

    for file in os.listdir(rf'{path}/labels'):
        if file.endswith('.npy'):
            labs.append(rf'{path}/labels/{file}')

    return np.asarray(ims), np.asarray(labs)

def shuffle(im, lab):
    p = np.random.permutation(len(im))

    return im[p], lab[p]