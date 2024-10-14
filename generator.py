import keras
import numpy as np
from utils import to_sparse
from tifffile import tifffile


class DataLoaderSWED(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, input_in_labels=False, biases=None) :
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.numpy = self.image_filenames[0].endswith('.npy')
        self.biases = biases
        if self.numpy:
            self.input_size = (np.moveaxis(np.load(image_filenames[0]), 0, 0)[:,:,1:4]).shape
        else:
            self.input_size = (np.moveaxis(tifffile.imread(image_filenames[0]), 0, -1)[:,:,1:4]).shape
        


    def __len__(self) :
        return (np.floor(len(self.image_filenames) / float(self.batch_size))).astype(int)


    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        images = []
        labels = []
        if self.numpy:
            for img_file, label_file in zip(batch_x, batch_y):
                images.append(np.moveaxis(np.load(img_file), 0, 0)[:,:,1:4].astype(np.float32)/22_000) #24_000
                labels.append(np.moveaxis(to_sparse(np.load(label_file)[0]),0,-1).astype(np.float32))
        else:
            for img_file, label_file in zip(batch_x, batch_y):
                images.append(np.moveaxis(tifffile.imread(img_file), 0, -1)[:,:,1:4].astype(np.float32)/22_000) #24_000    #SWED
                # images.append(np.moveaxis(tifffile.imread(img_file), 0, -1)[:,:,3:0:-1].astype(np.float32) / (10000*2.5277)) #SNOWED
                # images.append(np.moveaxis(tifffile.imread(img_file), 0, -1)[:,:,1:4].astype(np.float32) / (10000*2.5277))
                labels.append(np.moveaxis(to_sparse(tifffile.imread(label_file)),0,-1).astype(np.float32))

        if self.input_in_labels:
            return np.asarray(images), np.concatenate([np.asarray(labels), np.asarray(images)], axis=3)
        elif self.biases is not None:
            return (np.asarray(images), self.biases), np.asarray(labels)
        else:
            return np.asarray(images), np.asarray(labels)
    def get_all_labels(self):
        if self.numpy:
            return np.asarray([np.moveaxis(to_sparse(np.load(label)[0]),0,-1).astype(np.float32) for label in self.labels[:self.__len__()*self.batch_size]])
        else:
            return np.asarray([np.moveaxis(to_sparse(tifffile.imread(label)),0,-1).astype(np.float32) for label in self.labels[:self.__len__()*self.batch_size]])
        
class DataLoaderSNOWED(keras.utils.Sequence):

    def __init__(self, folder_names, batch_size, input_in_labels=False, root_path=rf'.\SNOWED\SNOWED', biases=None):
        self.folder_names = folder_names
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.root_path = root_path
        self.biases = biases
        self.input_size = (np.moveaxis(np.load(f'{root_path}/{folder_names[0]}/sample.npy'), 0, 0)[:,:,3:0:-1]).shape
        


    def __len__(self) :
        return (np.floor(len(self.folder_names) / float(self.batch_size))).astype(int)


    def __getitem__(self, idx) :
        batch = self.folder_names[idx * self.batch_size : (idx+1) * self.batch_size]

        images = []
        labels = []
        for folder_name in batch:
            images.append(np.moveaxis(np.load(f'{self.root_path}/{folder_name}/sample.npy'), 0, 0)[:,:,3:0:-1].astype(np.float32) / (10000*2.5277)) #24_000
            # images.append(np.moveaxis(np.load(f'{self.root_path}/{folder_name}/sample.npy'), 0, 0)[:,:,[1,2,3]].astype(np.float32) / (10000*2.5277)) #24_000
            
            labels.append(np.moveaxis(to_sparse(np.load(f'{self.root_path}/{folder_name}/label.npy')),0,-1).astype(np.float32))
        
        if self.input_in_labels:
            return np.asarray(images), np.concatenate([np.asarray(labels), np.asarray(images)], axis=3)
        elif self.biases is not None:
            return (np.asarray(images), self.biases), np.asarray(labels)
        else:
            return np.asarray(images), np.asarray(labels)
        
        

        
    def get_all_labels(self):
        return np.asarray([np.moveaxis(to_sparse(np.load(f'{self.root_path}/{folder_name}/label.npy')),0,-1) for folder_name in self.folder_names[:self.__len__()*self.batch_size]]).astype(np.float32)