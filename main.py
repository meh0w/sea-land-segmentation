import numpy as np
import DeepUNet
import DeepUNet2
import SeNet
import SeNet2

import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from generator import DataLoaderSWED, DataLoaderSNOWED
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff, accuracy, all_metrics
import wandb
from keras.callbacks import Callback
from losses import SeNet_loss, Sobel_loss, Sorensen_Dice, Weighted_Dice
from time import time

if __name__ == '__main__':
    try:
        MODEL = 'SeNet2'
        EPOCHS = 200
        BATCH_SIZE = 20
        LEARNING_RATE = 1e-4
        TRAIN_PART = 0.7

        BEST_WEIGHTS = None
        BEST_IOU = 0
        BEST_EPOCH = 0
        DEBUG = False
        DATASET = "SWED"
        LOSS = 'Weighted_Dice+Crossentropy'
        METRICS = 'ALL'

        if DATASET == 'SWED':
            # PATH = rf'.\SWED\train'
            PATH = rf'.\sample\train'
        elif DATASET == 'SNOWED':
            PATH = rf'.\SNOWED\SNOWED'
        
        class MetricsCallbackBatch(Callback):
            def on_epoch_end(self, epoch, logs=None):
                for i in range(len(data_loader_valid)):
                    res = self.model(data_loader_valid[i][0])[0]

                    pred = np.argmax(res, 3)
                    real = np.argmax(data_loader_valid[i][1], 3)
                    metrics_v = all_metrics(pred, real, "[VAL]")
                    if i == 0:
                        metrics_val = metrics_v
                        metrics_val['[VAL] loss (seg)'] = [loss['output_seg'](data_loader_valid[i][1], res).numpy()]
                        metrics_val['[VAL] loss (edge)'] = [loss['output_edge'](data_loader_valid[i][1], res).numpy()]
                    else:
                        metrics_val['[VAL] loss (seg)'] += [loss['output_seg'](data_loader_valid[i][1], res).numpy()]
                        metrics_val['[VAL] loss (edge)'] += [loss['output_edge'](data_loader_valid[i][1], res).numpy()]
                        for key in metrics_v:
                            metrics_val[key] = np.append(metrics_val[key], metrics_v[key])

                metrics_val = {key: np.mean(value) for key, value in metrics_val.items()}
                    

                for i in range(len(data_loader_train)):
                    res = self.model(data_loader_train[i][0])[0]

                    pred = np.argmax(res, 3)
                    real = np.argmax(data_loader_train[i][1], 3)
                    metrics_t = all_metrics(pred, real, "[TRAIN]")
                    if i == 0:
                        metrics_train = metrics_t
                    else:
                        for key in metrics_train:
                            metrics_train[key] = np.append(metrics_train[key], metrics_t[key])

                metrics_train = {key: np.mean(value) for key, value in metrics_train.items()}

                metrics = dict(metrics_val, **metrics_train)
                metrics['[TRAIN] loss'] = logs['loss']

                if not DEBUG:
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
                    os.makedirs(f'./weights/{MODEL}/27_12_2023', exist_ok=True)
                    self.model.save_weights(f'./weights/{MODEL}/27_12_2023/{metrics["[VAL] IoU mean"]:.6f}.h5')
                    # self.model.stop_training = True


        class MetricsCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):

                if type(loss) == dict:
                    res = self.model.predict(x=data_loader_valid)[0]

                    pred = np.argmax(res, 3)
                    real = np.argmax(labs_valid, 3)
                    metrics_val = all_metrics(pred, real, "[VAL]")
                    metrics_val['[VAL] loss (seg)'] = loss['output_seg'](labs_valid, res).numpy()
                    metrics_val['[VAL] loss (edge)'] = loss['output_edge'](labs_valid, res).numpy()
                else:
                    res = self.model.predict(x=data_loader_valid)

                    pred = np.argmax(res, 3)
                    real = np.argmax(labs_valid, 3)
                    metrics_val = all_metrics(pred, real, "[VAL]")
                    metrics_val['[VAL] loss'] = loss(labs_valid, res).numpy()

                metrics_val = {key: np.mean(value) for key, value in metrics_val.items()}
                    
                if type(loss) == dict:
                    res = self.model.predict(x=data_loader_train)[0]
                else:
                    res = self.model.predict(x=data_loader_train)

                pred = np.argmax(res, 3)
                real = np.argmax(labs_train, 3)
                metrics_train = all_metrics(pred, real, "[TRAIN]")

                metrics_train = {key: np.mean(value) for key, value in metrics_train.items()}

                metrics = dict(metrics_val, **metrics_train)
                metrics['[TRAIN] loss'] = logs['loss']

                if not DEBUG:
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
                    os.makedirs(f'./weights/{MODEL}/27_12_2023', exist_ok=True)
                    self.model.save_weights(f'./weights/{MODEL}/27_12_2023/{metrics["[VAL] IoU mean"]:.6f}.h5')
                    # self.model.stop_training = True

        if not DEBUG:
            # # # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="Masters thesis",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": LEARNING_RATE,
                "architecture": MODEL,
                "dataset": DATASET,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "loss": LOSS
                }
            )

        if DATASET == 'SWED':
            img_files, label_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_loader_train = DataLoaderSWED(img_files[idx], label_files[idx], BATCH_SIZE, False)
            data_loader_valid = DataLoaderSWED(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, False)

        elif DATASET == 'SNOWED':
            img_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_loader_train = DataLoaderSNOWED(img_files[idx], BATCH_SIZE, False)
            data_loader_valid = DataLoaderSNOWED(np.delete(img_files, idx), BATCH_SIZE, False)

        if MODEL == 'DeepUNet':
            m = DeepUNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'DeepUNet2':
            m = DeepUNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet':
            m = SeNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet2':
            m = SeNet2.get_model(data_loader_train.input_size, BATCH_SIZE)

        if METRICS == 'ALL':
            metrics_callback = MetricsCallback()
        elif METRICS == 'BATCH':
            metrics_callback = MetricsCallbackBatch()
        loss_weights = None
        if LOSS == 'Sorensen_Dice':
            loss = Sorensen_Dice()
        elif LOSS == 'CategoricalCrossentropy':
            loss = CategoricalCrossentropy()
        elif LOSS == 'Weighted_Dice':
            loss = Weighted_Dice()
        elif LOSS == 'Sobel':
            loss = Sobel_loss()
        elif LOSS == 'Sobel+Crossentropy':
            loss={'output_seg':CategoricalCrossentropy(),
                'output_edge':Sobel_loss()}
            loss_weights = {'output_seg': 1, 'output_edge':1}
        elif LOSS == 'Weighted_Dice+Crossentropy':
            loss={'output_seg':CategoricalCrossentropy(),
                'output_edge':Weighted_Dice()}
            loss_weights = {'output_seg': 1, 'output_edge':1}
        opt = Adam(learning_rate=LEARNING_RATE)

        labs_train = data_loader_train.get_all_labels()
        labs_valid = data_loader_valid.get_all_labels()

        if loss_weights is None:
            m.compile(optimizer=opt, loss=loss)
        else:
            m.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
        # m.run_eagerly = True
        print(f'WEIGHTS TO BE SAVED IN => ./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/BEST_DeepUNET.h5')
        m.fit(x=data_loader_train ,epochs=EPOCHS, callbacks=[metrics_callback])

        # res = m.predict(x=data_loader_train)

        # res_smax = res

        # predicted = np.argmax(res_smax, 3)
        # real = np.argmax(labs_train, 1)

        # fig, ax = plt.subplots(5, 2)
        # for i, im in enumerate(predicted):
        #     ax[i, 0].imshow(im)
        #     ax[i, 1].imshow(real[i])
        #     if i == 4:
        #         break

        plt.plot([1,2,3])
        plt.show()

        os.makedirs(f'./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/', exist_ok=True)
        m.save_weights(f'./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/LAST_DeepUNET.h5')

        m.set_weights(BEST_WEIGHTS)
        m.save_weights(f'./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/BEST_DeepUNET.h5')
        print('a')
    except KeyboardInterrupt:
        os.makedirs(f'./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/', exist_ok=True)
        m.save_weights(f'./weights/{MODEL}/26_12_2023 {LOSS} {DATASET} sample/LAST_DeepUNET.h5')

        m.set_weights(BEST_WEIGHTS)
        m.save_weights(f'./weights/{MODEL}/27_12_2023 {LOSS} {DATASET} sample/BEST_DeepUNET.h5')