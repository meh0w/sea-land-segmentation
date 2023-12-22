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
from losses import SeNet_loss, dummy

PATH = rf'.\sample\train'
EPOCHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 0.000001
TRAIN_PART = 0.7

BEST_WEIGHTS = None
BEST_IOU = 0
BEST_EPOCH = 0

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(x=data_loader_valid)

        pred = np.argmax(res, 3)
        real = np.argmax(labs_valid, 1)
        iou_valid = IoU(pred, real)

        res = self.model.predict(x=data_loader_train)

        pred = np.argmax(res, 3)
        real = np.argmax(labs_train, 1)
        iou_train = IoU(pred, real)
        # dice_ = dice_coeff(pred, real)
        
        wandb.log(
            {
            f'IoU mean val': np.mean(np.mean(iou_valid, axis=0)),
            f'IoU land val': np.mean(iou_valid[0]),
            f'IoU water val': np.mean(iou_valid[1]),

            f'IoU mean train': np.mean(np.mean(iou_train, axis=0)),
            f'IoU land train': np.mean(iou_train[0]),
            f'IoU water train': np.mean(iou_train[1]),
            f'Training loss': logs['loss'],
            f'Validation loss': logs['val_loss'],
            # f'Dice mean': np.mean(dice_),
            # f'Dice land': dice_[0],
            # f'Dice water': dice_[1]
             }
            )
        global BEST_IOU
        if np.mean(iou_valid) > BEST_IOU:
            global BEST_WEIGHTS, BEST_EPOCH
            BEST_WEIGHTS = self.model.get_weights()
            BEST_IOU = np.mean(iou_valid)
            BEST_EPOCH = epoch

        elif epoch - BEST_EPOCH > 50:
            self.model.stop_training = True

        if np.mean(iou_valid) > 0.95:
            os.makedirs(f'./weights/SeNet/02_11_2023', exist_ok=True)
            self.model.save_weights(f'./weights/SeNet/02_11_2023/{np.mean(iou_valid):.6f}.h5')
            # self.model.stop_training = True


# # # start a new wandb run to track this script
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


img_files, label_files = get_file_names(PATH, '.npy')
idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)

data_loader_train = DataLoader(img_files[idx], label_files[idx], BATCH_SIZE, True)
data_loader_valid = DataLoader(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, True)

m = get_model(data_loader_train.input_size, BATCH_SIZE)

metrics_callback = MetricsCallback()
# loss = CategoricalCrossentropy()
loss = SeNet_loss()
opt = Adam(learning_rate=LEARNING_RATE)
labs_train = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader_train.labels])
labs_valid = np.asarray([to_sparse(np.load(label)[0]) for label in data_loader_valid.labels])

m.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# dummy(data_loader_train[0][1], m(data_loader_train[0][0]))
m.fit(x=data_loader_train, validation_data=data_loader_valid ,epochs=EPOCHS, callbacks=[metrics_callback])

res = m.predict(x=data_loader_train)

# res_smax = Softmax(axis=-1)(res)
res_smax = res

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
os.makedirs(f'./weights/SeNet/02_11_2023', exist_ok=True)
m.save_weights(f'./weights/SeNet/02_11_2023/final_SeNet.h5')

m.set_weights(BEST_WEIGHTS)
m.save_weights(f'./weights/SeNet/02_11_2023/BEST_SeNet.h5')
print('a')