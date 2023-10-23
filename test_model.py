import numpy as np
from SeNet import get_model
import os
import matplotlib.pyplot as plt
from generator import DataLoader
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff


PATH = rf'.\sample\train'
BATCH_SIZE = 10
img_files, label_files = get_file_names(PATH)

data_loader = DataLoader(img_files, label_files, BATCH_SIZE)
m = get_model(data_loader.input_size, BATCH_SIZE)

res = m.predict(x=data_loader)
m.load_weights('./weights/0.8123.h5')
res_smax = Softmax(axis=-1)(res)
labs = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader.labels])

predicted = np.argmax(res_smax, 3)
real = np.argmax(labs, 1)

fig, ax = plt.subplots(5, 2)
for i, im in enumerate(predicted):
    ax[i, 0].imshow(im)
    ax[i, 1].imshow(real[i])
    if i == 4:
        break

iou = np.mean(IoU(predicted, real))
dice = np.mean(dice_coeff(predicted, real))
plt.show()