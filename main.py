import numpy as np
from DeepUNet import get_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import sklearn
from generator import DataLoader
from utils import get_file_names, to_sparse

PATH = rf'.\sample\train'
BATCH_SIZE = 15

img_files, label_files = get_file_names(PATH)

data_loader = DataLoader(img_files, label_files, BATCH_SIZE)

m = get_model(data_loader.input_size)

opt = Adam(learning_rate=0.0001, clipnorm=1.0, clipvalue=0.5)
m.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
m.fit(x=data_loader, epochs=5)

res = m.predict(x=data_loader)
labs = [to_sparse(np.load(label)[0]) for label in data_loader.labels]

fig, ax = plt.subplots(5)
for i, im in enumerate(res):
    ax[i].imshow(np.argmax(im, 0))
    if i == 4:
        break
plt.show()