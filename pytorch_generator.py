import os
from torch.utils.data import Dataset
from tifffile import tifffile
import numpy as np
from utils import to_sparse
import torch
import shutil

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
            # image = torch.from_numpy(np.moveaxis(tifffile.imread(self.image_filenames[idx]), 0, 0)[3:0:-1,:,:]/22_000).to(self.precision)
            image = torch.clamp(torch.from_numpy(tifffile.imread(self.image_filenames[idx])[[3,2,1],:,:].astype('int32')).to(self.precision)/22_000,min=0,max=1)
            # label = torch.from_numpy(np.moveaxis(to_sparse(tifffile.imread(self.labels[idx])),0,0)).to(self.precision)
            label = torch.nn.functional.one_hot(torch.from_numpy(tifffile.imread(self.labels[idx]).astype('int32')).to(torch.int64), 2).to(self.precision).permute(2,0,1)


        return image, label

    def get_all_labels(self):
        if self.numpy:
            return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(np.load(label)[0]),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision)
        else:
            return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(tifffile.imread(label)),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision)
        
class DataLoaderSWED_NDWI(DataLoaderSWED):
    def __init__(self, image_filenames, labels, input_in_labels=False, precision=32, inference=False):
        super().__init__(image_filenames, labels, input_in_labels=input_in_labels, precision=precision)

        if not inference:
            self.get_item = self.get_item_train
            self.get_len = self.get_len_train
        else:
            self.get_item = self.get_item_test
            self.get_len = self.get_len_test

    def get_len_train(self):
        return len(self.image_filenames)

    def get_len_test(self):
        return max(len(self.image_filenames), len(self.labels))

    def __len__(self):
        return self.get_len()

    def transform(self, im):
        im[3,:,:] = (im[1,:,:] - im[3,:,:] + 1e-9) / (im[1,:,:] + im[3,:,:] + 1e-9)
        return im

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item_train(self, idx):
        # image = torch.from_numpy(self.transform(tifffile.imread(self.image_filenames[idx])[[3,2,1,7],:,:]/22_000)).to(self.precision)
        # label = torch.from_numpy(np.moveaxis(to_sparse(tifffile.imread(self.labels[idx])),0,0)).to(self.precision)
        image = self.transform((torch.clamp(torch.from_numpy(tifffile.imread(self.image_filenames[idx])[[3,2,1,7],:,:].astype('int32')).to(self.precision)/22_000,min=0,max=1)))

        # label = torch.from_numpy(to_sparse(np.load(self.labels[idx])[0])).to(self.precision)
        label = torch.nn.functional.one_hot(torch.from_numpy(tifffile.imread(self.labels[idx]).astype('int32')).to(torch.int64), 2).to(self.precision).permute(2,0,1)

        return image, label

    def get_item_test(self, idx):
        image, label = None, None
        error = [False, False]
        if len(self.image_filenames) > idx:
            try:
                image = self.transform((torch.clamp(torch.from_numpy(tifffile.imread(self.image_filenames[idx])[[3,2,1,7],:,:].astype('int32')).to(self.precision)/22_000,min=0,max=1)))
            except:
                image = None
                error[0] = True
        if len(self.labels) > idx:
            try:
                label = torch.nn.functional.one_hot(torch.from_numpy(tifffile.imread(self.labels[idx]).astype('int32')).to(torch.int64), 2).to(self.precision).permute(2,0,1)
            except:
                label = None
                error[1] = True
        return image, label, error

    def get_all_labels(self):
        return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(tifffile.imread(label)),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision) 
  
class DataLoaderSWED_NDWI_np(DataLoaderSWED):
    def __init__(self, image_filenames, labels, input_in_labels=False, precision=32):
        # idx = []
        # for i, lab in enumerate(labels):         
        #     if sorted(np.unique(np.load(lab)[0])) == [0,1] or sorted(np.unique(np.load(lab)[0])) == [0] or sorted(np.unique(np.load(lab)[0])) == [1]:
        #         continue
        #     else:
        #         shutil.move(image_filenames[i], rf'C:\Users\Michal\Documents\INF\MAG\SWED_incorrect')
        #         shutil.move(lab, rf'C:\Users\Michal\Documents\INF\MAG\SWED_incorrect')
        #         # idx.append(i)      
        # image_filenames = np.delete(image_filenames, idx)
        # labels = np.delete(labels, idx)
        super().__init__(image_filenames, labels, input_in_labels=input_in_labels, precision=precision)

    def transform(self, im):
        im[3,:,:] = (im[1,:,:] - im[3,:,:] + 1e-9) / (im[1,:,:] + im[3,:,:] + 1e-9)
        return im
    def __getitem__(self, idx):
        # image = torch.from_numpy(self.transform(np.moveaxis(np.load(self.image_filenames[idx]), 2, 0)[[3,2,1,7],:,:]/22_000)).to(self.precision)
        image = self.transform((torch.clamp(torch.from_numpy(np.load(self.image_filenames[idx])[:,:,[3,2,1,7]]).to(self.precision)/22_000,min=0,max=1)).permute((2,0,1)))

        # label = torch.from_numpy(to_sparse(np.load(self.labels[idx])[0])).to(self.precision)
        label = torch.nn.functional.one_hot(torch.from_numpy(np.load(self.labels[idx])[0]).to(torch.int64), 2).permute((2,0,1)).to(self.precision)

        return image, label

    def get_all_labels(self):
        return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(np.load(label)[0]),0,0) for label in self.labels[:self.__len__()*self.batch_size]])).to(self.precision)
        
class DataLoaderSNOWED(Dataset):

    def __init__(self, folder_names, batch_size, input_in_labels=False, root_path=rf'.\SNOWED\SNOWED', precision=32):
        self.folder_names = folder_names
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.root_path = root_path
        self.input_size = (np.moveaxis(np.load(f'{root_path}/{folder_names[0]}/sample.npy'), 0, 0)[3:0:-1,:,:]).shape
        if precision == 32:
            self.precision = torch.float32
        elif precision == 16:
            self.precision = torch.float16
        
    def __len__(self) :
        return len(self.folder_names)


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
    
class DataLoaderSNOWED_NDWI(Dataset):

    def __init__(self, folder_names, batch_size, input_in_labels=False, root_path=rf'.\SNOWED\SNOWED', precision=32):
        self.folder_names = folder_names
        self.batch_size = batch_size
        self.input_in_labels = input_in_labels
        self.root_path = root_path
        self.input_size = (np.moveaxis(np.load(f'{root_path}/{folder_names[0]}/sample.npy'), 0, 0)[[3,2,1,7],:,:]).shape
        if precision == 32:
            self.precision = torch.float32
        elif precision == 16:
            self.precision = torch.float16
    def __len__(self) :
        return len(self.folder_names)
    def transform(self, im):
        im[3,:,:] = (im[1,:,:] - im[3,:,:] + 1e-9) / (im[1,:,:] + im[3,:,:] + 1e-9)
        return im

    def __getitem__(self, idx) :

        images = []
        labels = []
        image = torch.from_numpy(self.transform(np.moveaxis(np.load(f'{self.root_path}/{self.folder_names[idx]}/sample.npy'), 2, 0)[[3,2,1,7],:,:].astype(np.float32) / (10000*2.5277)))#24_000
        # label = torch.from_numpy(np.moveaxis(to_sparse(np.load(f'{self.root_path}/{self.folder_names[idx]}/label.npy')),0,0).astype(np.float32))
        # torch.nn.functional.one_hot(torch.from_numpy(np.load(self.labels[idx])[0]).to(torch.int64), 2).permute((2,0,1)).to(self.precision)
        label = torch.nn.functional.one_hot(torch.from_numpy(np.load(f'{self.root_path}/{self.folder_names[idx]}/label.npy')).to(torch.int64), 2).permute((2,0,1)).to(self.precision)
        
        if self.input_in_labels:
            return torch.from_numpy(images), torch.cat([torch.from_numpy(labels), torch.from_numpy(images)], dim=3)
        else:
            return image, label
        
    def get_all_labels(self):
        return torch.from_numpy(np.asarray([np.moveaxis(to_sparse(np.load(f'{self.root_path}/{folder_name}/label.npy')),0,0) for folder_name in self.folder_names[:self.__len__()*self.batch_size]]).astype(np.float32))