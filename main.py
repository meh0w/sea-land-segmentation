import numpy as np
from DeepUNet import get_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from generator import DataLoader
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff, accuracy, all_metrics
import wandb
from keras.callbacks import Callback
from losses import SeNet_loss, dummy, Sobel_loss, Sorensen_Dice, Sorensen_Dice2, Weighted_Dice, Weighted_Dice2

PATH = rf'.\sample\train'
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 0.000001
TRAIN_PART = 0.7

BEST_WEIGHTS = None
BEST_IOU = 0
BEST_EPOCH = 0
DEBUG = False

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        res = self.model.predict(x=data_loader_valid)

        pred = np.argmax(res, 3)
        real = np.argmax(labs_valid, 1)
        metrics_val = all_metrics(pred, real, "[VAL]")

        res = self.model.predict(x=data_loader_train)

        pred = np.argmax(res, 3)
        real = np.argmax(labs_train, 1)
        metrics_train = all_metrics(pred, real, "[TRAIN]")
        metrics = dict(metrics_val, **metrics_train)
        metrics['[TRAIN] loss'] = logs['loss']
        metrics['[VAL] loss'] = logs['val_loss']

        wandb.log(
            metrics
            )
        global BEST_IOU
        if metrics["[VAL] IoU mean"] > BEST_IOU:
            global BEST_WEIGHTS, BEST_EPOCH
            BEST_WEIGHTS = self.model.get_weights()
            BEST_IOU = metrics["[VAL] IoU mean"]
            BEST_EPOCH = epoch

        elif epoch - BEST_EPOCH > 50:
            self.model.stop_training = True

        if metrics["[VAL] IoU mean"] > 0.95:
            os.makedirs(f'./weights/DeepUNet/24_12_2023', exist_ok=True)
            self.model.save_weights(f'./weights/DeepUNet/24_12_2023/{metrics["[VAL] IoU mean"]:.6f}.h5')
            # self.model.stop_training = True


if not DEBUG:
    # # # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Masters thesis",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "DeepUNet",
        "dataset": "SWED",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
        }
    )

img_files, label_files = get_file_names(PATH, '.npy')
idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)

data_loader_train = DataLoader(img_files[idx], label_files[idx], BATCH_SIZE, False)
data_loader_valid = DataLoader(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, False)

m = get_model(data_loader_train.input_size, BATCH_SIZE)

metrics_callback = MetricsCallback()
# loss = CategoricalCrossentropy()
loss = Sorensen_Dice()
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

plt.show()
os.makedirs(f'./weights/SeNet/02_11_2023', exist_ok=True)
m.save_weights(f'./weights/SeNet/02_11_2023/final_SeNet.h5')

m.set_weights(BEST_WEIGHTS)
m.save_weights(f'./weights/SeNet/02_11_2023/BEST_SeNet.h5')
print('a')