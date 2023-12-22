import numpy as np
from SeNet import get_model
import os
import matplotlib.pyplot as plt
from generator import DataLoader
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff
from tifffile import tifffile


PATH = rf'.\sample\train'
BATCH_SIZE = 1
img_files, label_files = get_file_names(PATH, '.npy')

data_loader = DataLoader(img_files, label_files, BATCH_SIZE)
idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*0.7)), replace=False)
# data_loader = DataLoader(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, False)
m = get_model(data_loader.input_size, BATCH_SIZE)

m.load_weights('./weights/SeNet/02_11_2023/BEST_SeNet.h5')
# m.load_weights('./weights/DeepUNet/02_11_2023/BEST_DeepUNet.h5')
res = m.predict(x=data_loader)
# res_smax = Softmax(axis=-1)(res)
if label_files[0].endswith('.npy'):
    labs = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader.labels])
    # images = np.asarray([np.moveaxis(np.load(img_file), 0, 0)[:,:,:]/22_000 for img_file in data_loader.image_filenames])
else:
    labs = np.asarray([to_sparse(tifffile.imread(label)) for label in data_loader.labels])
    # images = np.asarray([np.moveaxis(tifffile.imread(img_file), 0, -1)[:,:,:]/22_000 for img_file in data_loader.image_filenames])

predicted = np.argmax(res, 3)
real = np.argmax(labs, 1)

number_of_rows = min(5, len(predicted))
fig, ax = plt.subplots(number_of_rows, 3)
j = 0
predicted.shape

data_eval = []

for i, im in enumerate(predicted):
    iou = np.nanmean(IoU(im, real[i]))
    data_eval.append({'idx':i, 'iou':iou})

data_eval = sorted(data_eval, key=lambda x: x['iou'])
test = np.mean(np.nanmean(IoU(predicted, real), axis=0))
print(f'TEST IoU: {test}')
print(len(img_files))
for entry in data_eval:
    i = entry['idx']
    print(entry['iou'])
# for i in range(len(predicted)):
    if number_of_rows > 1:
        rgb = data_loader[i][0][:,:,:,[0,1,2]][0]
        ax[j, 0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
        ax[j, 1].imshow(predicted[i])
        ax[j, 2].imshow(real[i])

        ax[j, 0].set_xticks([])
        ax[j, 0].set_yticks([])

        ax[j, 1].set_xticks([])
        ax[j, 1].set_yticks([])

        ax[j, 2].set_xticks([])
        ax[j, 2].set_yticks([])
        
    else:
        rgb = data_loader[i][0][:,:,:,[0,1,2]][0]
        ax[0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
        ax[1].imshow(predicted[i])
        ax[2].imshow(real[i])

        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].set_xticks([])
        ax[2].set_yticks([])

    j += 1
    if j == number_of_rows:
        plt.subplots_adjust(0.021, 0.012, 0.376, 0.998, 0, 0.067)
        plt.show()
        fig, ax = plt.subplots(5, 3)
        j = 0
    print(j)

iou = np.mean(IoU(predicted, real))
dice = np.mean(dice_coeff(predicted, real))
print(iou)

# fig, ax = plt.subplots(3)
# ax[0].imshow((images[i][:,:,:3]-np.min(images[i][:,:,:3]))/(np.max(images[i][:,:,:3])-np.min(images[i][:,:,:3])))
# ax[1].imshow(im)
# ax[2].imshow(real[i])
# plt.show()