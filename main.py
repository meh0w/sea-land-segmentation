import numpy as np
import DeepUNet
import DeepUNet2
import SeNet
import SeNet2
import MU_Net

import os
# import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from generator import DataLoaderSWED, DataLoaderSNOWED
from utils import get_file_names, to_sparse, add_weight_decay
from tensorflow.keras.layers import Softmax
from metrics import IoU, dice_coeff, accuracy, all_metrics
import wandb
from keras.callbacks import Callback, LearningRateScheduler
from losses import SeNet_loss, Sobel_loss, Sorensen_Dice, Weighted_Dice,CEDice
from time import time, sleep
from datetime import datetime
from tensorflow import stack
import tensorflow as tf

def poly_scheduler(epoch, lr):
    return 0.001*((1-(epoch/150))**0.9)

def custom_scheduler(epoch, lr):
    if epoch < 30:
        return 0.001*((1-(epoch/30))**0.9)
    else:
        return (0.001*((1-(29/30))**0.9))*((1-(epoch/150))**0.9)

def saturate_scheduler(epoch, lr):
    if epoch < 30:
        return 0.001*((1-(epoch/30))**0.9)
    else:
        return (0.001*((1-(29/30))**0.9))

def run(c):
    try:
        MODEL = c['MODEL']
        EPOCHS = c['EPOCHS']
        BATCH_SIZE = c['BATCH_SIZE']
        LEARNING_RATE = c['LEARNING_RATE']
        TRAIN_PART = c['TRAIN_PART']

        DEBUG = c['DEBUG']
        DATASET = c['DATASET']
        LOSS = c['LOSS']
        METRICS = c['METRICS']
        SCHEDULER = c['SCHEDULER']
        OPTIMIZER = c['OPTIMIZER']
        MOMENTUM = c['MOMENTUM']
        WEIGHT_DECAY = c['WEIGHT_DECAY']

        if DATASET == 'SWED':
            PATH = rf'.\sample\train'
        elif DATASET == 'SNOWED':
            PATH = rf'.\SNOWED\SNOWED'
        elif DATASET == 'SWED_FULL':
            PATH = rf'.\SWED\train'
        
        class MetricsCallbackBatch(Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print("a")
            def on_epoch_end(self, epoch, logs=None):
                for i in range(len(data_loader_valid)):
                    res = self.model(data_loader_valid[i][0])

                    pred = np.argmax(res, 3)
                    real = np.argmax(data_loader_valid[i][1], 3)
                    metrics_v = all_metrics(pred, real, "[VAL]")
                    if type(loss) == dict:
                        if i == 0:
                            metrics_val = metrics_v
                            metrics_val['[VAL] loss (seg)'] = [loss['output_seg'](data_loader_valid[i][1], res).numpy()]
                            metrics_val['[VAL] loss (edge)'] = [loss['output_edge'](data_loader_valid[i][1], res).numpy()]
                        else:
                            metrics_val['[VAL] loss (seg)'] += [loss['output_seg'](data_loader_valid[i][1], res).numpy()]
                            metrics_val['[VAL] loss (edge)'] += [loss['output_edge'](data_loader_valid[i][1], res).numpy()]
                            for key in metrics_v:
                                metrics_val[key] = np.append(metrics_val[key], metrics_v[key])
                    else:
                        if i == 0:
                            metrics_val = metrics_v
                            metrics_val['[VAL] loss'] = [loss(data_loader_valid[i][1], res).numpy()]
                        else:
                            metrics_val['[VAL] loss'] += [loss(data_loader_valid[i][1], res).numpy()]
                            for key in metrics_v:
                                metrics_val[key] = np.append(metrics_val[key], metrics_v[key])

                metrics_val = {key: np.mean(value) for key, value in metrics_val.items()}
                    

                for i in range(len(data_loader_train)):
                    res = self.model(data_loader_train[i][0])

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

                elif epoch - BEST_EPOCH > 20:
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

                elif epoch - BEST_EPOCH > 20:
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
                "loss": LOSS,
                "scheduler": SCHEDULER,
                "optimizer": OPTIMIZER,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY
                }
            )
        biases = None
        if MODEL == 'MU_Net':
            b1 = MU_Net.get_bias()
            b2 = MU_Net.get_bias()
            b3 = MU_Net.get_bias()
            b4 = MU_Net.get_bias()
            b5 = MU_Net.get_bias()
            b6 = MU_Net.get_bias()
            biases = stack([b1,b2,b3,b4,b5,b6])

        if DATASET == 'SWED':
            img_files, label_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_loader_train = DataLoaderSWED(img_files[idx], label_files[idx], BATCH_SIZE, False, biases)
            data_loader_valid = DataLoaderSWED(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, False, biases)

        elif DATASET == 'SNOWED':
            img_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_loader_train = DataLoaderSNOWED(img_files[idx], BATCH_SIZE, False, biases=biases)
            data_loader_valid = DataLoaderSNOWED(np.delete(img_files, idx), BATCH_SIZE, False, biases=biases)

        elif DATASET == 'SWED_FULL':
            img_files, label_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_loader_train = DataLoaderSWED(img_files[idx], label_files[idx], BATCH_SIZE, False, biases)
            data_loader_valid = DataLoaderSWED(np.delete(img_files, idx), np.delete(label_files, idx), BATCH_SIZE, False, biases)

        if MODEL == 'DeepUNet':
            m = DeepUNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'DeepUNet2':
            m = DeepUNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet':
            m = SeNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet2':
            m = SeNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'MU_Net':
            b1 = MU_Net.get_bias()
            b2 = MU_Net.get_bias()
            b3 = MU_Net.get_bias()
            b4 = MU_Net.get_bias()
            b5 = MU_Net.get_bias()
            b6 = MU_Net.get_bias()
            biases = stack([b1,b2,b3,b4,b5,b6])
            print(biases.shape)
            m = MU_Net.get_model(data_loader_train.input_size, biases.shape, batch_size=BATCH_SIZE)

        callbacks = []
        if METRICS == 'ALL':
            metrics_callback = MetricsCallback()
        elif METRICS == 'BATCH':
            metrics_callback = MetricsCallbackBatch()
        callbacks.append(metrics_callback)
        if SCHEDULER == 'poly':
            scheduler_callback = LearningRateScheduler(poly_scheduler)
            callbacks.append(scheduler_callback)
        elif SCHEDULER == 'custom':
            scheduler_callback = LearningRateScheduler(custom_scheduler)
            callbacks.append(scheduler_callback)
        elif SCHEDULER == 'saturate':
            scheduler_callback = LearningRateScheduler(saturate_scheduler)
            callbacks.append(scheduler_callback) 
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
        elif LOSS == 'Dice+Crossentropy':
            loss = CEDice()
        if OPTIMIZER == 'Adam':
            opt = Adam(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == 'SGD':
            opt = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
            add_weight_decay(m, WEIGHT_DECAY)
        elif OPTIMIZER == 'RMSProp':
            opt = RMSprop(learning_rate=LEARNING_RATE)

        if METRICS == 'ALL':
            labs_train = data_loader_train.get_all_labels()
            labs_valid = data_loader_valid.get_all_labels()

        if loss_weights is None:
            m.compile(optimizer=opt, loss=loss)
        else:
            m.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
        # m.run_eagerly = True
        now = datetime.now()
        RES_PATH_FOLDER = f'./weights/{MODEL}/{now.strftime("%Y-%m-%d %H_%M_%S")} {LOSS} {DATASET} {LEARNING_RATE:.0e} sample'
        RES_PATH = f'{RES_PATH_FOLDER}/BEST.h5'
        print(f'WEIGHTS TO BE SAVED IN => {RES_PATH}')

        m.fit(x=data_loader_train ,epochs=EPOCHS, callbacks=callbacks)

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

        # plt.plot([1,2,3])
        # plt.show()

        os.makedirs(RES_PATH_FOLDER, exist_ok=True)
        m.save_weights(rf'{RES_PATH_FOLDER}\LAST.h5')

        m.set_weights(BEST_WEIGHTS)
        m.save_weights(RES_PATH)
        with open(f'{RES_PATH_FOLDER}/config.txt', 'w+') as f:
            print(c, file=f)
        
        if biases is not None:
            np.save(rf'{RES_PATH_FOLDER}\biases', biases.numpy())
        print('a')
        wandb.finish()
    except KeyboardInterrupt:
        os.makedirs(RES_PATH_FOLDER, exist_ok=True)
        m.save_weights(rf'{RES_PATH_FOLDER}\LAST.h5')

        m.set_weights(BEST_WEIGHTS)
        m.save_weights(RES_PATH)
        with open(f'{RES_PATH_FOLDER}/config.txt', 'w+') as f:
            print(c, file=f)
        if biases is not None:
            np.save(rf'{RES_PATH_FOLDER}\biases', biases.numpy())


if __name__ == '__main__':

    configs = [
    #     {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "saturate",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "RMSProp",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    ###############SWED
    ##OPTIMIZERS
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "saturate",
    #     "DEBUG":  True,
    #     "DATASET": "SWED_FULL",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'BATCH',
    #     "OPTIMIZER": "SGD",
    #     "MOMENTUM": 0.9,
    #     "WEIGHT_DECAY": 1e-4
    # },

    {
        "MODEL": 'MU_Net',
        "EPOCHS":  3,
        "BATCH_SIZE": 10,
        "LEARNING_RATE":  1e-3,
        "TRAIN_PART":  0.7,
        "SCHEDULER": "saturate",
        "DEBUG":  True,
        "DATASET": "SWED",
        "LOSS": 'Dice+Crossentropy',
        "METRICS": 'BATCH',
        "OPTIMIZER": "Adam",
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4
    },
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "saturate",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "Adam",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  1,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "saturate",
    #     "DEBUG":  True,
    #     "DATASET": "SNOWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "RMSProp",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    # #SCHEDULERS
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "poly",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "Adam",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "custom",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "Adam",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "poly",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "RMSProp",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    # {
    #     "MODEL": 'MU_Net',
    #     "EPOCHS":  150,
    #     "BATCH_SIZE": 10,
    #     "LEARNING_RATE":  1e-3,
    #     "TRAIN_PART":  0.7,
    #     "SCHEDULER": "custom",
    #     "DEBUG":  False,
    #     "DATASET": "SWED",
    #     "LOSS": 'Dice+Crossentropy',
    #     "METRICS": 'ALL',
    #     "OPTIMIZER": "RMSProp",
    #     "MOMENTUM": 0,
    #     "WEIGHT_DECAY": 0
    # },
    ]
    for cc in configs:
        print(f'{cc["MODEL"]}| {cc["DATASET"]} | {cc["LOSS"]} | {cc["LEARNING_RATE"]}')
    sleep(5)
    

    for configuration in configs:
        BEST_WEIGHTS = None
        BEST_IOU = 0
        BEST_EPOCH = 0
        run(configuration)
        break
        