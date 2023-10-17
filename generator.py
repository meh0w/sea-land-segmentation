import keras
import numpy as np
from utils import to_sparse


class DataLoader(keras.utils.all_utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

        self.input_size = (np.moveaxis(np.load(image_filenames[0]), -1, 0)[:,:,:]/24_000).shape


    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        images = []
        labels = []
        for img_file, label_file in zip(batch_x, batch_y):
            images.append(np.moveaxis(np.load(img_file), -1, 0)[:,:,:]/24_000)
            labels.append(to_sparse(np.load(label_file)[0]))
        return np.asarray(images), np.asarray(labels)