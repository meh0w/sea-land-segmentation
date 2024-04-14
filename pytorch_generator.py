import os
from torch.utils.data import Dataset
from tifffile import tifffile
import numpy as np
from utils import to_sparse
import torch

class DataLoaderSWED(Dataset):
    def __init__(self, image_filenames, labels, input_in_labels=False, precision=32):
        self.image_filenames = image_filenames
        self.labels = labels
        self.input_in_labels = input_in_labels
        self.numpy = self.image_filenames[0].endswith('.npy')
        if precision == 32:
            self.precision = torch.float32
        elif precision == 16:
            self.precision = torch.float16
        if self.numpy:
            self.input_size = (np.moveaxis(np.load(image_filenames[0]), 0, 0)[:,:,1:4]).shape
        else:
            self.input_size = (np.moveaxis(tifffile.imread(image_filenames[0]), 0, -1)[:,:,1:4]).shape

    def __len__(self) :
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.numpy:
            image = torch.from_numpy(np.moveaxis(np.load(self.image_filenames[idx]), 2, 0)[3:0:-1,:,:]/22_000).to(self.precision)
            label = torch.from_numpy(np.moveaxis(to_sparse(np.load(self.labels[idx])[0]),0,0)).to(self.precision)
        else:
            image = torch.from_numpy(np.moveaxis(tifffile.imread(self.image_filenames[idx]), 0, 0)[3:0:-1,:,:]/22_000).to(self.precision)
            label = torch.from_numpy(np.moveaxis(to_sparse(tifffile.imread(self.labels[idx])),0,0)).to(self.precision)

        return image, label

    def get_all_labels(self):
        if self.numpy:
            return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(np.load(label)[0]),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision)
        else:
            return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(tifffile.imread(label)),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision)
class DataLoaderSNOWED(Dataset):

    def __init__(self, folder_names, batch_size, input_in_labels=False, root_path=rf'.\SNOWED\SNOWED'):
        self.folder_names = folder_names
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.root_path = root_path
        self.input_size = (np.moveaxis(np.load(f'{root_path}/{folder_names[0]}/sample.npy'), 0, 0)[3:0:-1,:,:]).shape
        
    def __len__(self) :
        return (np.floor(len(self.folder_names) / float(self.batch_size))).astype(int)


    def __getitem__(self, idx) :
        batch = self.folder_names[idx * self.batch_size : (idx+1) * self.batch_size]

        images = []
        labels = []
        for folder_name in batch:
            images.append(np.moveaxis(np.load(f'{self.root_path}/{folder_name}/sample.npy'), 2, 0)[3:0:-1,:,:].astype(np.float32) / (10000*2.5277)) #24_000
            labels.append(np.moveaxis(to_sparse(np.load(f'{self.root_path}/{folder_name}/label.npy')),0,0).astype(np.float32))
        
        images = np.asarray(images)
        labels = np.asarray(labels)
        if self.input_in_labels:
            return torch.from_numpy(images), torch.cat([torch.from_numpy(labels), torch.from_numpy(images)], dim=3)
        else:
            return torch.from_numpy(images), torch.from_numpy(labels)
        
    def get_all_labels(self):
        return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(np.load(f'{self.root_path}/{folder_name}/label.npy')),0,0) for folder_name in self.folder_names[:self.__len__()*self.batch_size]]).astype(np.float32))