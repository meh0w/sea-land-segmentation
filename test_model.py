import numpy as np
import DeepUNet
import DeepUNet2
import SeNet
import SeNet2
import os
import matplotlib.pyplot as plt
from generator import DataLoaderSWED, DataLoaderSNOWED
from utils import get_file_names, to_sparse
from tensorflow.keras.layers import Softmax
from tensorflow import stack
from metrics import IoU, dice_coeff, all_metrics, ConfusionMatrix
from tifffile import tifffile
from pprint import pprint
import MU_Net
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

def test(data_path, weights_path, data_loader, m, display, save, result_path):
    # PATH = rf'.\SWED\test'
    BATCH_SIZE = 1

    m.load_weights(weights_path)
    # m.load_weights('./weights/DeepUNet/02_11_2023/BEST_DeepUNet.h5')
    res = m.predict(x=data_loader)
    if type(res) == list:
        res = res[0]
    # res_smax = Softmax(axis=-1)(res)
    labs = data_loader.get_all_labels()

    predicted = np.argmax(res, 3)
    real = np.argmax(labs, 3)

    number_of_rows = min(6, len(predicted))
    fig, ax = plt.subplots(number_of_rows, 3)
    j = 0
    predicted.shape

    data_eval = []
    conf_matrix = ConfusionMatrix()
    for i, im in enumerate(predicted):
        metrics_test = {key: np.mean(value) for key, value in all_metrics(im, real[i]).items()}
        data_eval.append({'idx':i, 'iou':metrics_test[' IoU mean']})

        conf_matrix.calc(im, real[i])

    IoU_land, IoU_water, IoU_mean = conf_matrix.get_IoU_new()
    metrics_test[' IoU_land_new'] = IoU_land
    metrics_test[' IoU_water_new'] = IoU_water
    metrics_test[' IoU mean_new'] = IoU_mean
    data_eval = sorted(data_eval, key=lambda x: x['iou'])
    test = {key: np.mean(value) for key, value in all_metrics(predicted, real).items()}
    test[' IoU_land_new'] = IoU_land[0]
    test[' IoU_water_new'] = IoU_water[0]
    test[' IoU mean_new'] = IoU_mean[0]
    # pprint(f'TEST metrics: {test}')
    with open(rf'{result_path}.txt', 'w') as f:
        for key, value in test.items():
            print(f'{key}: {np.round(value, 4):.4f} \n')
            f.write(f'{key}: {np.round(value, 4):.4f} \n')
    # return
    *_, last1, last2,last3 = data_eval
    # print(len(img_files))
    for entry in [data_eval[0],data_eval[1],data_eval[2], last1, last2,last3]:
        i = entry['idx']
        print(entry['iou'])
    # for i in range(len(predicted)):
        if number_of_rows > 1:
            if isinstance(data_loader, DataLoaderSWED):
                # rgb = data_loader[i][0][0][:,:,:,[2, 1,0]][0]
                if data_loader.biases is None:
                    rgb = data_loader[i][0][0][:,:,[2, 1,0]]
                else:
                    rgb = data_loader[i][0][0][0][:,:,[2, 1,0]]
            else:
                if data_loader.biases is None:
                    rgb = data_loader[i][0][0][:,:,:,[0,1,2]][0]
                else:
                    rgb = data_loader[i][0][0][0][:,:,[2, 1,0]]
            rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
            clipped = np.clip(rgb, 0, 0.6)/0.6
            # clipped = rgb
            
            # clipped = rgb
            # ax[j, 0].hist(rgb[:,:,0].flatten(), bins=100)
            # ax[j, 1].hist(rgb[:,:,1].flatten(), bins=100)
            # ax[j, 2].hist(rgb[:,:,2].flatten(), bins=100)
            # ax[j, 0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
            ax[j, 0].imshow(clipped)
            ax[j, 1].imshow(predicted[i])
            ax[j, 2].imshow(real[i])

            ax[j, 0].set_xticks([])
            ax[j, 0].set_yticks([])

            ax[j, 1].set_xticks([])
            ax[j, 1].set_yticks([])

            ax[j, 2].set_xticks([])
            ax[j, 2].set_yticks([])
            
        else:
            rgb = data_loader[i][0][:,:,:,[0,1,2]][0]
            # ax[0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
            ax[0].imshow((rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb)))
            ax[1].imshow(predicted[i])
            ax[2].imshow(real[i])

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

    # metrics = all_metrics()

# fig, ax = plt.subplots(3)
# ax[0].imshow((images[i][:,:,:3]-np.min(images[i][:,:,:3]))/(np.max(images[i][:,:,:3])-np.min(images[i][:,:,:3])))
# ax[1].imshow(im)
# ax[2].imshow(real[i])
# plt.show()
    
def test_one_folder(folder, result_folder=None, test_set=""):
    # 1
    # folder = rf'weights\DeepUNet\24_12_2023 Sorensen_Dice SNOWED'
    # 2
    # folder = rf'weights\DeepUNet\24_12_2023 Weighted_Dice SNOWED'
    # folder = rf'weights\DeepUNet\2023-12-29 04_53_30 Weighted_Dice SNOWED 1e-06 sample' # no batch norm
    # # 3
    # folder = rf'weights\DeepUNet2\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample'
    # # 4
    # folder = rf'weights\DeepUNet\26_12_2023 Sorensen_Dice SWED sample'
    # # 5
    # folder = rf'weights\DeepUNet\26_12_2023 Weighted_Dice SWED sample'
    # # 6
    # folder = rf'weights\DeepUNet2\26_12_2023 Weighted_Dice+Crossentropy SWED sample'
    # folder = rf'weights\DeepUNet2\2023-12-28 22_11_25 Weighted_Dice+Crossentropy SWED 1e-04 sample' #redone
    # # 7
    # folder = rf'weights\DeepUNet2\2023-12-29 03_33_39 Weighted_Dice+Crossentropy SWED 1e-06 sample'
    # # 8
    # folder = rf'weights\SeNet\27_12_2023 Sorensen_Dice SNOWED sample'
    # # 9
    # folder = rf'weights\SeNet\27_12_2023 Weighted_Dice SNOWED sample'
    # # 10
    # folder = rf'weights\SeNet2\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample'
    # # 11
    # folder = rf'weights\SeNet\27_12_2023 Sorensen_Dice SWED sample'
    # # 12
    # folder = rf'weights\SeNet\27_12_2023 Weighted_Dice SWED sample'
    # # 13
    # folder = rf'weights\SeNet2\27_12_2023 Weighted_Dice+Crossentropy SWED sample'
    # 14
    # folder = rf'weights\SeNet2\2023-12-29 02_11_31 Weighted_Dice+Crossentropy SWED 1e-06 sample'

    ## MU_NET
    # folder = rf'weights\MU_Net\2024-03-04 09_44_05 CategoricalCrossentropy SWED 1e-06 sample'

    BATCH_SIZE = 1
    # folder = rf'weights\SeNet2\27_12_2023 Weighted_Dice+Crossentropy SWED sample'
    weights = rf'.\{folder}\BEST_DeepUNET.h5'

    
    
    biases = None
    if 'MU_Net' in folder:
        if os.path.isfile(rf'.\{folder}\biases.npy'):
            biases = np.load(rf'.\{folder}\biases.npy')
        else:
            print("WARNING: BIASES FOR MU_NET NOT FOUND, CREATING NEW BIASES EXPECTED WORSE RESULTS")
            b1 = MU_Net.get_bias()
            b2 = MU_Net.get_bias()
            b3 = MU_Net.get_bias()
            b4 = MU_Net.get_bias()
            b5 = MU_Net.get_bias()
            b6 = MU_Net.get_bias()
            biases = stack([b1,b2,b3,b4,b5,b6])

    if ('SWED' in folder and test_set != 'SNOWED') or test_set == 'SWED':
        data = rf'.\SWED\test'
        img_files, label_files = get_file_names(data, '.tif', 'SWED')
        loader = DataLoaderSWED(img_files, label_files, BATCH_SIZE, False, biases=biases)

    elif 'SNOWED' in folder or test_set == 'SNOWED':
        data = rf'.\SNOWED_TEST'
        img_files = get_file_names(data, '.npy', 'SNOWED')
        loader = DataLoaderSNOWED(img_files, BATCH_SIZE, False, root_path=data,biases=biases)

    # if 'SNOWED' in folder:
    # data = rf'.\SNOWED\SNOWED'
    # img_files = get_file_names(data, '.npy', 'SNOWED')
    # loader2 = DataLoaderSNOWED(img_files, BATCH_SIZE, False)
    # elif 'SWED' in folder:
    #     data = rf'.\SWED\test'
    #     loader = DataLoaderSWED
    #     img_files, label_files = get_file_names(data, '.npy', 'SNOWED')
    #     loader = DataLoaderSWED(img_files, label_files, BATCH_SIZE, False)
    # fig, ax = plt.subplots(2, 2)
    # rgb = loader[3][0][0][:,:,[2, 1,0]]
    # rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
    # ax[0, 0].imshow(rgb)
    # ax[0, 1].imshow(np.argmax(loader[3][1][0], 2))

    # rgb = loader2[0][0][0]
    # rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
    # ax[1, 0].imshow(rgb)
    # ax[1, 1].imshow(np.argmax(loader2[0][1][0], 2))
    # plt.show()

    if 'DeepUNet2' in folder:
        model = DeepUNet2.get_model(loader.input_size, BATCH_SIZE)
    elif 'DeepUNet' in folder:
        model = DeepUNet.get_model(loader.input_size, BATCH_SIZE)
    elif 'SeNet2' in folder:
        model = SeNet2.get_model(loader.input_size, BATCH_SIZE)
    elif 'SeNet' in folder:
        model = SeNet.get_model(loader.input_size, BATCH_SIZE)
    elif 'MU_Net' in folder:
        model = MU_Net.get_model(loader.input_size, biases.shape, batch_size=BATCH_SIZE)
    
    save = True
    display = False
    if save and result_folder is None:
        raise Exception('Results folder not specified')
    else:
        file = folder.split('\\')[-1]
        result_path = rf"{result_folder}\{file}"
        test(data, weights, loader, model, display, save, result_path)

if __name__ == '__main__':
    base = rf'weights\MU_Net\REPORT 09-03-2024'

    #Tab 5.2
    # folders = {
    # #1.
    # 1: rf'weights\DeepUNet\24_12_2023 Sorensen_Dice SNOWED',
    # #2.
    # # folder = rf'weights\DeepUNet\24_12_2023 Weighted_Dice SNOWED'
    # 2: rf'weights\DeepUNet\2023-12-29 04_53_30 Weighted_Dice SNOWED 1e-06 sample',
    # #3.
    # 3: rf'weights\DeepUNet2\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample',
    # #4.
    # 4: rf'weights\DeepUNet\26_12_2023 Sorensen_Dice SWED sample',
    # #5.
    # 5: rf'weights\DeepUNet\26_12_2023 Weighted_Dice SWED sample',
    # #6.
    # # folder = rf'weights\DeepUNet2\2023-12-28 22_11_25 Weighted_Dice+Crossentropy SWED 1e-04 sample'
    # 6: rf'weights\DeepUNet2\26_12_2023 Weighted_Dice+Crossentropy SWED sample',
    # #7.
    # 7: rf'weights\DeepUNet2\2023-12-29 03_33_39 Weighted_Dice+Crossentropy SWED 1e-06 sample',
    # #8.
    # 8: rf'weights\SeNet\27_12_2023 Sorensen_Dice SNOWED sample',
    # #9.?
    # 9: rf'weights\SeNet\27_12_2023 Weighted_Dice SNOWED sample',
    # #10.?~
    # 10: rf'weights\SeNet2\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample',
    # #11.?
    # 11: rf'weights\SeNet\27_12_2023 Sorensen_Dice SWED sample',
    # #12.?~
    # 12: rf'weights\SeNet\27_12_2023 Weighted_Dice SWED sample',
    # #13.?
    # 13: rf'weights\SeNet2\27_12_2023 Weighted_Dice+Crossentropy SWED sample',
    # #14.?~
    # 14: rf'weights\SeNet2\2023-12-29 02_11_31 Weighted_Dice+Crossentropy SWED 1e-06 sample',
    # }
    #Tab 5.5
    folders = {
    #1.
    1:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 12_53_30 Dice+Crossentropy SWED 1e-03 sample',
    #2.
    2:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 15_11_23 Dice+Crossentropy SWED 1e-03 sample',
    #3.
    3:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 17_24_06 Dice+Crossentropy SWED 1e-03 sample',
    #4.
    4:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 19_23_00 Dice+Crossentropy SWED 1e-03 sample',
    #5.
    5:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 22_21_17 Dice+Crossentropy SWED 1e-03 sample',
    #6.
    6:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 23_11_14 Dice+Crossentropy SWED 1e-03 sample',
    #7.
    7:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-09 01_42_36 Dice+Crossentropy SWED 1e-03 sample',
    #8.
    8:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-06 03_13_09 Dice+Crossentropy SNOWED 1e-03 sample',
    #9.
    9:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-06 09_23_00 Dice+Crossentropy SNOWED 1e-03 sample',
    #10.
    10:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-06 14_07_07 Dice+Crossentropy SNOWED 1e-03 sample',
    #11.
    11:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-07 02_56_47 Dice+Crossentropy SNOWED 1e-03 sample',
    #12.
    12:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-07 12_48_20 Dice+Crossentropy SNOWED 1e-03 sample',
    #13.
    13:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-07 17_00_55 Dice+Crossentropy SNOWED 1e-03 sample',
    #14.
    14:rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 04_29_27 Dice+Crossentropy SNOWED 1e-03 sample',
    }
    #NDWI
    #1.
    # folder rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #2.
    # folder = rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #3.
    # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #4.
    # folder = rf'weights\MU_Net\2024-04-16 12_50_21 Dice+Crossentropy SWED_FULL 1e-03 sample'

    # folder = rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 12_53_30 Dice+Crossentropy SWED 1e-03 sample'
    # folder = rf'weights\MU_Net\REPORT 09-03-2024\2024-03-08 12_53_30 Dice+Crossentropy SWED 1e-03 sample'
    # folder = rf'weights\SeNet2\2023-12-29 02_11_31 Weighted_Dice+Crossentropy SWED 1e-06 sample'
    # folder = rf'weights\DeepUNet\26_12_2023 Sorensen_Dice SWED sample'
    # folder = rf'weights\MU_Net\REPORT 09-03-2024\2024-03-07 02_56_47 Dice+Crossentropy SNOWED 1e-03 sample' #SNOWED_Adam_(2)
    for i, folder in folders.items():
        res = rf'plots\NEW_IOU\tab55#{i}'
        os.makedirs(res, exist_ok=True)
        test_one_folder(folder, res)
    # for folder in os.listdir(base):
    #     test_one_folder(rf'{base}\{folder}', res)
    #     break