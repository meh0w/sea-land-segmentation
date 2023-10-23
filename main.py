import numpy as np
from SeNet import get_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import sklearn
from generator import DataLoader
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff
import wandb
from keras.callbacks import Callback

PATH = rf'.\sample\train'
EPOCHS = 50
BATCH_SIZE = 15
LEARNING_RATE = 0.000001
TRAIN_PART = 0.7

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(x=data_loader_valid)

        res_smax = Softmax(axis=-1)(res)

        pred = np.argmax(res_smax, 3)
        real = np.argmax(labs_valid, 1)
        iou_ = IoU(pred, real)
        dice_ = dice_coeff(pred, real)
        
        wandb.log(
            {
            f'IoU mean': np.mean(iou_),
            f'IoU land': iou_[0],
            f'IoU water': iou_[1],
            f'Dice mean': np.mean(dice_),
            f'Dice land': dice_[0],
            f'Dice water': dice_[1]
             }
            )
        
        if np.mean(iou_) > 0.95:
            self.model.save_weights(f'./weights/{np.mean(iou_):.4f}.h5')
            self.model.stop_training = True


# # start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Masters thesis",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "SeNet",
    "dataset": "SWED",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE
    }
)


img_files, label_files = get_file_names(PATH)
idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*0.7)), replace=False)

data_loader_train = DataLoader(img_files[idx], label_files[idx], BATCH_SIZE)
data_loader_valid = DataLoader(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE)

m = get_model(data_loader_train.input_size, BATCH_SIZE)

metrics_callback = MetricsCallback()
loss = CategoricalCrossentropy(from_logits=True)
opt = Adam(learning_rate=LEARNING_RATE)
labs_train = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader_train.labels])
labs_valid = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader_valid.labels])

m.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
m.fit(x=data_loader_train, validation_data=data_loader_valid ,epochs=EPOCHS, callbacks=[metrics_callback])

res = m.predict(x=data_loader_train)

res_smax = Softmax(axis=-1)(res)

predicted = np.argmax(res_smax, 3)
real = np.argmax(labs_train, 1)

fig, ax = plt.subplots(5, 2)
for i, im in enumerate(predicted):
    ax[i, 0].imshow(im)
    ax[i, 1].imshow(real[i])
    if i == 4:
        break

iou = np.mean(IoU(predicted, real))
dice = np.mean(dice_coeff(predicted, real))
plt.show()
print('a')