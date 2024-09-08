import torch
import matplotlib.pyplot as plt
from pytorch_MU_NET import MUNet as MU_Net
import pytorch_MU_NET_quant as experimental
from pytorch_metrics import All_metrics
from pytorch_generator import DataLoaderSNOWED, DataLoaderSWED, DataLoaderSWED_NDWI,DataLoaderSWED_NDWI_np
import os
import pprint
from tifffile import tifffile
from utils import get_file_names
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

def quantize(model,device):
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # model = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])
    model_fp32_prepared = torch.ao.quantization.prepare(model)
    img_files, label_files = get_file_names(rf'.\SWED\train', '.npy', 'SWED_FULL')
    idx = np.random.choice(np.arange(len(img_files)), int(np.floor(len(img_files)*0.7)), replace=False)
    data_set_valid = DataLoaderSWED_NDWI_np(np.delete(img_files, idx), np.delete(label_files, idx), False, precision=32)
    data_loader_valid = DataLoader(data_set_valid, batch_size=10, shuffle=True, pin_memory=True, num_workers=2)
    for i, data in enumerate(data_loader_valid):
        img, label = data
        model_fp32_prepared(img.to(device))
        break
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared.to('cpu'))
    model_int8 = model_int8.to(device)

    return model_int8


def test(data_path, weights_path, data_loader, dataset, m, display, save, result_path, device='cpu'):
    m.eval()
    if QUANT:
        # m = quantize(m,device)
        m = torch.ao.quantization.quantize_dynamic(
            m,  # the original model
            {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
            dtype=torch.qint8)

    test_metrics = All_metrics(device, '[TEST]')
    softmax = torch.nn.Softmax(dim=1)
    val_epoch_progress = tqdm(
        data_loader, f"[EVAL] (TEST SET)", leave=False
    )
    with torch.no_grad():
            param_size = 0
            for param in m.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in m.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2
            print('model size: {:.3f}MB'.format(size_all_mb))
            times = []
            for i, data in enumerate(val_epoch_progress):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                start = time()
                outputs = m(inputs)
                times.append(time()-start)
                test_metrics.calc(outputs, labels)

            print(f'Czas inferencji: {np.mean(times[1:])}\n {times}')
            test_state_dict = test_metrics.get(False)
            data_eval = []
            metrics_test = {key: value.cpu().numpy() for key, value in test_state_dict.items()}
            for i in range(len(val_epoch_progress)):
                data_eval.append({'idx':i, 'iou':metrics_test['[TEST] IoU mean'][i]})
            data_eval = sorted(data_eval, key=lambda x: x['iou'])
            *_, last1, last2,last3 = data_eval
            if save:
                metrics = test_metrics.get(True)
                with open(rf'{result_path}.txt', 'w') as f:
                    for key, value in metrics.items():
                        print(f'{key}: {np.round(value, 4):.4f} \n')
                        f.write(f'{key}: {np.round(value, 4):.4f} \n')
            else:
                metrics = test_metrics.get(True)
                for key, value in metrics.items():
                    print(f'{key}: {np.round(value, 4):.4f} \n')
            print('a')

    number_of_rows = min(6, len(val_epoch_progress))
    fig, ax = plt.subplots(number_of_rows, 3)
    j = 0
    for entry in [data_eval[0],data_eval[1],data_eval[2], last1, last2,last3]:
        i = entry['idx']
        print(entry['iou'])
    # for i in range(len(predicted)):
        if isinstance(dataset, DataLoaderSWED):
            rgb = np.moveaxis(dataset[i][0].numpy(), 0, -1)[:,:,[0,1,2]]
            real = np.moveaxis(dataset[i][1].numpy(), 0, -1)
        else:
            rgb = dataset[i][0][0][:,:,:,[0,1,2]][0]
        rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
        clipped = np.clip(rgb, 0, 0.6)/0.6
        # clipped = rgb
        with torch.no_grad():
            predicted = softmax(m(dataset[i][0].to(device).unsqueeze(0)))
        if number_of_rows > 1:

            # clipped = rgb
            # ax[j, 0].hist(rgb[:,:,0].flatten(), bins=100)
            # ax[j, 1].hist(rgb[:,:,1].flatten(), bins=100)
            # ax[j, 2].hist(rgb[:,:,2].flatten(), bins=100)
            # ax[j, 0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
            ax[j, 0].imshow(clipped)
            ax[j, 1].imshow(np.argmax(predicted.cpu().numpy()[0],axis=0))
            ax[j, 2].imshow(np.argmax(real,axis=-1))

            ax[j, 0].set_xticks([])
            ax[j, 0].set_yticks([])

            ax[j, 1].set_xticks([])
            ax[j, 1].set_yticks([])

            ax[j, 2].set_xticks([])
            ax[j, 2].set_yticks([])
            
        else:
            # ax[0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
            ax[0].imshow(clipped)
            ax[1].imshow(np.argmax(predicted.cpu().numpy()[0],axis=0))
            ax[2].imshow(np.argmax(real,axis=-1))

            ax[0].set_xticks([])
            ax[0].set_yticks([])

            ax[1].set_xticks([])
            ax[1].set_yticks([])

            ax[2].set_xticks([])
            ax[2].set_yticks([])

        j += 1
        if j == number_of_rows:
            # plt.subplots_adjust(0.021, 0.012, 0.376, 0.998, 0, 0.067)
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            if save:
                fig.set_size_inches(2.4,4.75)
                plt.savefig(f'{result_path}.jpg', dpi=600)
            if display:
                plt.show()
            fig, ax = plt.subplots(5, 3)
            j = 0
        print(j)

def test_one_folder(folder, result_folder=None):
    BATCH_SIZE = 1
    weights = rf'.\{folder}\BEST.pt'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if QUANT:
        device = 'cpu'

    if 'DeepUNet' in folder:
        m = DeepUNet.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'DeepUNet2' in folder:
        m = DeepUNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'SeNet' in folder:
        m = SeNet.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'SeNet2' in folder:
        m = SeNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'MU_Net' in folder:
        # m = MU_Net([32,64,128,256], base_c=32, bilinear=False)
        m = MU_Net([4,64,128,256,512]) #1st NDWI
        # m = MU_Net([4, 32,64,128,256], base_c=32)
        # m = experimental.MUNet([4, 32,64,128,256], base_c=32)

        # m = MU_Net()
        # m = experimental.MUNet(encoder_channels=[4,32,64,128,256], base_c = 16)
        m.to(device)
        m.load_state_dict(torch.load(weights))
        print(f'Total params: {sum(p.numel() for p in m.parameters())}')
        print(f'Trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad)}')
        

    if 'SWED' in folder:
        PATH = rf'.\SWED\test'
        img_files, label_files = get_file_names(PATH, '.tif', 'SWED')
        # data_set_test = DataLoaderSWED(img_files, label_files, False)
        data_set_test = DataLoaderSWED_NDWI(img_files, label_files, False)

    elif 'SNOWED' in folder:
        PATH = rf'.\SNOWED_TEST'
        img_files = get_file_names(PATH, '.npy', 'SNOWED')
        data_set_test = DataLoaderSNOWED(img_files, False)

    elif 'SWED_FULL' in folder:
        PATH = rf'.\SWED\test'
        img_files, label_files = get_file_names(PATH, '.tif', 'SWED')
        data_set_test = DataLoaderSWED(img_files, label_files, False)

    loader = DataLoader(data_set_test, batch_size=BATCH_SIZE, shuffle=False)

    save = False
    display = True
    if save and result_folder is None:
        raise Exception('Results folder not specified')
    else:
        file = folder.split('\\')[-1]
        result_path = rf"{result_folder}\{file}"
        test(PATH, weights, loader, data_set_test, m, display, save, result_path, device)


if __name__ == '__main__':
    base = rf'weights\MU_Net\REPORT 20-04-2024'
    res = rf'plots\OSIOSN'
    os.makedirs(res, exist_ok=True)
    QUANT = True
    # folder = rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample' #1)?
    folder = rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample' #2)? #OSIOSN #1
    # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample' #3)?
    # folder = rf'weights\MU_Net\2024-04-16 12_50_21 Dice+Crossentropy SWED_FULL 1e-03 sample' #4)?
    # folder = rf'weights\MU_Net\2024-05-25 13_45_24 Dice+Crossentropy SWED 1e-03 sample'
    # folder = rf'weights\MU_Net\2024-05-30 15_16_40 Dice+Crossentropy SWED_FULL 1e-03 sample'

    #1st NDWI
    # folder = rf'weights\MU_Net\2024-04-14 17_55_05 Dice+Crossentropy SWED 1e-03 sample'

    # BEST FULL SWED
    # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample' #OSIOSN #2

    # folder = rf'weights\MU_Net\2024-05-27 01_38_49 Dice+Crossentropy SWED_FULL 1e-03 sample'

    # folder = rf'weights\MU_Net\2024-05-28 03_00_37 Dice+Crossentropy SWED_FULL 1e-03 sample'

    # no split
    # folder = rf'weights\MU_Net\2024-05-29 01_55_09 Dice+Crossentropy SWED_FULL 1e-03 sample'

    test_one_folder(folder, res)
    # for folder in os.listdir(base):
    #     test_one_folder(rf'{base}\{folder}', res)