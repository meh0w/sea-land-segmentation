import torch
import matplotlib.pyplot as plt
from pytorch_MU_NET import MUNet as MU_Net
from pytorch_metrics import All_metrics
from pytorch_generator import DataLoaderSNOWED, DataLoaderSWED, DataLoaderSWED_NDWI
import os
import pprint
from tifffile import tifffile
from utils import get_file_names
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def test(data_path, weights_path, data_loader, dataset, m, display, save, result_path, device='cpu'):
    m.eval()
    test_metrics = All_metrics(device, '[TEST]')
    softmax = torch.nn.Softmax(dim=1)
    val_epoch_progress = tqdm(
        data_loader, f"[EVAL] (TEST SET)", leave=False
    )
    with torch.no_grad():
            for i, data in enumerate(val_epoch_progress):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = m(inputs)
                test_metrics.calc(outputs, labels)

            test_state_dict = test_metrics.get(False)
            data_eval = []
            metrics_test = {key: value.cpu().numpy() for key, value in test_state_dict.items()}
            for i in range(len(val_epoch_progress)):
                data_eval.append({'idx':i, 'iou':metrics_test['[TEST] IoU mean'][i]})
            data_eval = sorted(data_eval, key=lambda x: x['iou'])
            *_, last1, last2,last3 = data_eval
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
            if display:
                plt.show()
            if save:
                fig.set_size_inches(2.4,4.75)
                plt.savefig(f'{result_path}.jpg', dpi=600)
            fig, ax = plt.subplots(5, 3)
            j = 0
        print(j)

def test_one_folder(folder, result_folder=None):
    BATCH_SIZE = 10
    weights = rf'.\{folder}\BEST.pt'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        m = MU_Net([4,64,128,256,512])
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
    base = rf'weights\MU_Net\REPORT 09-03-2024'
    res = rf'plots\pytorch_test2'
    os.makedirs(res, exist_ok=True)
    folder = rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample'
    test_one_folder(folder, res)
    # for folder in os.listdir(base):
    #     test_one_folder(rf'{base}\{folder}', res)