import keras
import numpy as np
from utils import to_sparse
from tifffile import tifffile


class DataLoader(keras.utils.all_utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, input_in_labels=False) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.numpy = self.image_filenames[0].endswith('.npy')
        if self.numpy:
            self.input_size = (np.moveaxis(np.load(image_filenames[0]), 0, 0)[:,:,1:4]).shape
        else:
            self.input_size = (np.moveaxis(tifffile.imread(image_filenames[0]), 0, -1)[:,:,1:4]).shape
        


    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)


    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        images = []
        labels = []
        if self.numpy:
            for img_file, label_file in zip(batch_x, batch_y):
                images.append(np.moveaxis(np.load(img_file), 0, 0)[:,:,1:4]/22_000) #24_000
                labels.append(np.moveaxis(to_sparse(np.load(label_file)[0]),0,-1))
        else:
            for img_file, label_file in zip(batch_x, batch_y):
                images.append(np.moveaxis(tifffile.imread(img_file), 0, -1)[:,:,1:4]/22_000) #24_000
                labels.append(np.moveaxis(to_sparse(tifffile.imread(label_file)),0,-1))

        if self.input_in_labels:
            return np.asarray(images), np.concatenate([np.asarray(labels), np.asarray(images)], axis=3)
        else:
            return np.asarray(images), np.asarray(labels)