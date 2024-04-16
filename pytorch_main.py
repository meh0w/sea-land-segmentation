import torch
import wandb
import os
from pytorch_MU_NET import MUNet as MU_Net
from pytorch_generator import DataLoaderSNOWED, DataLoaderSWED, DataLoaderSWED_NDWI, DataLoaderSWED_NDWI_np
import numpy as np
from utils import get_file_names
from datetime import datetime
from time import time, sleep
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_metrics import All_metrics, All_metrics_16
from pytorch_losses import CEDice
from copy import deepcopy

def poly_scheduler(epoch):
    return 0.001*((1-(epoch/150))**0.9)

def custom_scheduler(epoch):
    if epoch < 30:
        return 0.001*((1-(epoch/30))**0.9)
    else:
        return (0.001*((1-(29/30))**0.9))*((1-(epoch/150))**0.9)

def saturate_scheduler(epoch):
    if epoch < 30:
        return 0.001*((1-(epoch/30))**0.9)
    else:
        return (0.001*((1-(29/30))**0.9))


def run(c):
    try:
        BEST_WEIGHTS = None
        BEST_IOU = 0
        BEST_EPOCH = 0
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        NOTE = c['NOTE']
        PRECISION = c['PRECISION']
        NDWI = c['NDWI']

        if DATASET == 'SWED':
            PATH = rf'.\sample\train'
        elif DATASET == 'SNOWED':
            PATH = rf'.\SNOWED\SNOWED'
        elif DATASET == 'SWED_FULL':
            PATH = rf'.\SWED\train'
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
                "weight_decay": WEIGHT_DECAY,
                "note": NOTE,
                "NDWI": NDWI
                }
            )

        if DATASET == 'SWED':
            if NDWI:
                img_files, label_files = get_file_names(PATH, '.npy', DATASET)
                idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
                data_set_train = DataLoaderSWED_NDWI_np(img_files[idx], label_files[idx], False, precision=PRECISION)
                data_set_valid = DataLoaderSWED_NDWI_np(np.delete(img_files, idx), np.delete(label_files, idx), False, precision=PRECISION)

            else:
                img_files, label_files = get_file_names(PATH, '.npy', DATASET)
                idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
                data_set_train = DataLoaderSWED(img_files[idx], label_files[idx], False, precision=PRECISION)
                data_set_valid = DataLoaderSWED(np.delete(img_files, idx), np.delete(label_files, idx), False, precision=PRECISION)

        elif DATASET == 'SNOWED':
            img_files = get_file_names(PATH, '.npy', DATASET)
            idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
            data_set_train = DataLoaderSNOWED(img_files[idx], False)
            data_set_valid = DataLoaderSNOWED(np.delete(img_files, idx),False)

        elif DATASET == 'SWED_FULL':
            if NDWI:
                img_files, label_files = get_file_names(PATH, '.npy', DATASET)
                idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
                data_set_train = DataLoaderSWED_NDWI_np(img_files[idx], label_files[idx], False, precision=PRECISION)
                data_set_valid = DataLoaderSWED_NDWI_np(np.delete(img_files, idx), np.delete(label_files, idx), False, precision=PRECISION)

            else:
                img_files, label_files = get_file_names(PATH, '.npy', DATASET)
                idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*TRAIN_PART)), replace=False)
                data_set_train = DataLoaderSWED(img_files[idx], label_files[idx], False, precision=PRECISION)
                data_set_valid = DataLoaderSWED(np.delete(img_files, idx), np.delete(label_files, idx), False, precision=PRECISION)

        data_loader_train = DataLoader(data_set_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        data_loader_valid = DataLoader(data_set_valid, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

        if MODEL == 'DeepUNet':
            m = DeepUNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'DeepUNet2':
            m = DeepUNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet':
            m = SeNet.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'SeNet2':
            m = SeNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
        elif MODEL == 'MU_Net':
            # m = MU_Net([32,64,128,256], base_c=32, bilinear=True)
            m = MU_Net(encoder_channels=[4,32,64,128,256], base_c = 32)
            m.to(device)
            if PRECISION == 16:
                m.half()

        ## RESULTS FILE
        now = datetime.now()
        RES_PATH_FOLDER = f'./weights/{MODEL}/{now.strftime("%Y-%m-%d %H_%M_%S")} {LOSS} {DATASET} {LEARNING_RATE:.0e} sample'
        RES_PATH = f'{RES_PATH_FOLDER}/BEST.h5'
        print(f'WEIGHTS TO BE SAVED IN => {RES_PATH}')
        if SCHEDULER == 'poly':
            scheduler_fn = poly_scheduler
        elif SCHEDULER == 'custom':
            scheduler_fn = custom_scheduler
        elif SCHEDULER == 'saturate':
            scheduler_fn = saturate_scheduler
        else:
            scheduler_fn = None
        
        if LOSS == 'Crossentropy':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif LOSS == 'Dice+Crossentropy':
            loss_fn = CEDice()

    ##Training loop
        if OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(m.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        elif OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
        if scheduler_fn is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        else:
            scheduler = None

        if PRECISION == 32:
            train_metrics = All_metrics(device, '[TRAIN]')
            val_metrics = All_metrics(device, '[VAL]')
        elif PRECISION == 16:
            train_metrics = All_metrics_16(device, '[TRAIN]')
            val_metrics = All_metrics_16(device, '[VAL]')

        main_progress_bar = tqdm(
            range(EPOCHS), desc="Training progress", position=0
        )

        for epoch in main_progress_bar:
            main_progress_bar.set_postfix(Epoch=f"{epoch} / {EPOCHS}")
            train_epoch_progress = tqdm(
                data_loader_train, f"Epoch {epoch} (Train)", leave=False
            )
            m.train()
            running_loss = 0.
            train_metrics.clear()
            val_metrics.clear()
            for i, data in enumerate(train_epoch_progress):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = m(inputs)
                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                train_epoch_progress.set_postfix(
                    Loss=f"{running_loss / (1+i):.4f}",
                )
            train_epoch_progress.close()
            m.eval()
            with torch.no_grad():
                train_loss = 0
                train_epoch_progress = tqdm(
                    data_loader_train, f"[EVAL] Epoch {epoch} (Train)", leave=False
                )
                for i, data in enumerate(train_epoch_progress):
                    inputs, labels = data
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = m(inputs)
                    train_metrics.calc(outputs, labels)
                    loss = loss_fn(outputs, labels)
                    train_loss += loss.item()
                    train_epoch_progress.set_postfix(
                    Loss=f"{train_loss / (1+i):.4f}",
                    )
                train_state_dict = train_metrics.get()
                train_state_dict['[TRAIN] loss'] = train_loss / len(train_epoch_progress)
                val_loss = 0
                val_epoch_progress = tqdm(
                    data_loader_valid, f"[EVAL] Epoch {epoch} (Valid)", leave=False
                )
                for i, data in enumerate(val_epoch_progress):
                    inputs, labels = data
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = m(inputs)
                    loss = loss_fn(outputs, labels)
                    val_metrics.calc(outputs, labels)
                    val_loss += loss.item()
                    val_epoch_progress.set_postfix(
                    Loss=f"{val_loss / (1+i):.4f}",
                    )
                val_state_dict = val_metrics.get()
                val_state_dict['[VAL] loss'] = val_loss / len(val_epoch_progress)

                metrics = dict(train_state_dict, **val_state_dict)
                if not DEBUG:
                    wandb.log(
                        metrics
                        )
                if metrics["[VAL] IoU mean"] > BEST_IOU:
                    BEST_WEIGHTS = deepcopy(m.state_dict())
                    BEST_IOU = metrics["[VAL] IoU mean"]
                    BEST_EPOCH = epoch

                elif epoch - BEST_EPOCH > 10:
                    # Early stopping
                    main_progress_bar.close()
                    os.makedirs(RES_PATH_FOLDER, exist_ok=True)
                    torch.save(m.state_dict(), rf'{RES_PATH_FOLDER}\LAST.pt')
                    torch.save(BEST_WEIGHTS, rf'{RES_PATH_FOLDER}\BEST.pt')
                    with open(f'{RES_PATH_FOLDER}/config.txt', 'w+') as f:
                        print(c, file=f)
                    wandb.finish()
                    return
            if scheduler is not None:
                scheduler.step()
        main_progress_bar.close()
        os.makedirs(RES_PATH_FOLDER, exist_ok=True)
        torch.save(m.state_dict(), rf'{RES_PATH_FOLDER}\LAST.pt')
        torch.save(BEST_WEIGHTS, rf'{RES_PATH_FOLDER}\BEST.pt')
        with open(f'{RES_PATH_FOLDER}/config.txt', 'w+') as f:
            print(c, file=f)
        wandb.finish()

    except KeyboardInterrupt:
        # Manual interruption
        os.makedirs(RES_PATH_FOLDER, exist_ok=True)
        torch.save(m.state_dict(), rf'{RES_PATH_FOLDER}\LAST.pt')
        torch.save(BEST_WEIGHTS, rf'{RES_PATH_FOLDER}\BEST.pt')
        with open(f'{RES_PATH_FOLDER}/config.txt', 'w+') as f:
            print(c, file=f)
        wandb.finish()

if __name__ == '__main__':

    configs = [
        {
        "MODEL": 'MU_Net',
        "EPOCHS":  100,
        "BATCH_SIZE": 10,
        "LEARNING_RATE":  1e-3,
        "TRAIN_PART":  0.7,
        "SCHEDULER": "poly",
        "DEBUG":  False,
        "DATASET": "SWED_FULL",
        "LOSS": 'Dice+Crossentropy',
        "METRICS": 'BATCH',
        "OPTIMIZER": "Adam",
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 1e-4,
        "NOTE": 'softmax in loss no weight init encoder_channels=[4,32,64,128,256], base_c = 32',
        "PRECISION": 32,
        "NDWI": True,
        },
    ]
    for cc in configs:
        print(f'{cc["MODEL"]}| {cc["DATASET"]} | {cc["LOSS"]} | {cc["LEARNING_RATE"]}')
    sleep(5)


    for configuration in configs:
        run(configuration)
        break