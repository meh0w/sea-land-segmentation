import torch
import matplotlib.pyplot as plt
from pytorch_DeepUNet import DeepUNet
from pytorch_SeNet import SeNet
from pytorch_MU_NET import MUNet as MU_Net
import pytorch_MU_NET_quant as experimental
# from pytorch_metrics import All_metrics
from pytorch_metrics_new import All_metrics
from pytorch_generator import DataLoaderSNOWED, DataLoaderSWED, DataLoaderSWED_NDWI,DataLoaderSWED_NDWI_np, DataLoaderSNOWED_NDWI
import os
import pprint
from tifffile import tifffile
from utils import get_file_names
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import torch.nn.utils.prune as prune
from fasterai.misc.bn_folding import BN_Folder
import ast

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.pt")
    print()
    size = f"Size (MB): {os.path.getsize('temp.pt')/1e6}"
    os.remove('temp.pt')
    return size


def test(data_loader, dataset, m, plot_settings, results_folder, results_file, options, device='cpu'):
    m.eval()

    save = plot_settings['save']
    display = plot_settings['display']
    no_predictions = plot_settings['no_predictions']
    separated = plot_settings['separated']

    with torch.no_grad():
        if options['bn fold']:
            bn = BN_Folder()
            m = bn.fold(m)
            print('BN FOLDED')

    if options['prune']:
        parameters_to_prune = (
        (m.cnn_encoder.down1.doubleconv.double_conv[0], 'weight'),
        (m.cnn_encoder.down1.doubleconv.double_conv[3], 'weight'),
        (m.cnn_encoder.down2.doubleconv.double_conv[0], 'weight'),
        (m.cnn_encoder.down2.doubleconv.double_conv[3], 'weight'),
        (m.cnn_encoder.down3.conv[0], 'weight'),
        (m.cnn_encoder.down4.conv[0], 'weight'),
        (m.trans_decoder.up1.conv[0], 'weight'),
        (m.trans_decoder.up2.conv.double_conv[0], 'weight'),
        (m.trans_decoder.up2.conv.double_conv[3], 'weight'),
        (m.trans_decoder.up3.conv.double_conv[0], 'weight'),
        (m.trans_decoder.up3.conv.double_conv[3], 'weight'),
        (m.trans_decoder.up4.conv.double_conv[0], 'weight'),
        (m.trans_decoder.up4.conv.double_conv[3], 'weight'),
        )

        prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=options['prune amount'],
        )
        
        for param in parameters_to_prune:
            prune.remove(param[0], 'weight')
        print('PRUNED')

    with open(rf'{results_folder}/metrics_{results_file}.txt', 'w') as f:
        f.write(f'Total params: {sum(p.numel() for p in m.parameters())} \n')
        f.write(f'Trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad)} \n')

    if options['quant']:
        m = torch.ao.quantization.quantize_dynamic(
            m,  # the original model
            {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
            dtype=options['quant dtype'])
        print('QUANTIZED')


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
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    test_metrics.calc(outputs[0], labels)
                else:
                    test_metrics.calc(outputs, labels)

            print(f'Czas inferencji: {np.mean(times[1:])}\n {times}')
            test_state_dict = test_metrics.get(False)
            data_eval = []
            metrics_test = {key: value.cpu().numpy() for key, value in test_state_dict.items()}
            for i in range(len(val_epoch_progress)):
                data_eval.append({
                    'idx':i, 
                    'iou':metrics_test['[TEST] IoU mean'][i],
                    'iou_water':metrics_test['[TEST] IoU water'][i],
                    'iou_land':metrics_test['[TEST] IoU land'][i]
                    })
            data_eval = sorted(data_eval, key=lambda x: x['iou'])
            *_, last1, last2,last3 = data_eval
            if save:
                metrics = test_metrics.get(True)
                with open(rf'{results_folder}/metrics_{results_file}.txt', 'a') as f:
                    for key, value in metrics.items():
                        print(f'{key}: {np.round(value, 4):.4f} \n')
                        f.write(f'{key}: {np.round(value, 4):.4f} \n')
                    f.write(f'Inference time: {np.mean(times[1:])}\n')
                    f.write(f'{print_size_of_model(m)}')
                    f.write(f'Optimization options: {options} \n')

            else:
                metrics = test_metrics.get(True)
                for key, value in metrics.items():
                    print(f'{key}: {np.round(value, 4):.4f} \n')
            print('a')

    if separated:
        fig_img, ax_img = plt.subplots(1, 1, figsize=(1, 1))
        fig_pred, ax_pred = plt.subplots(1, 1, figsize=(1, 1))
        fig_label, ax_label = plt.subplots(1, 1, figsize=(1, 1))
        for row_num, entry in enumerate([data_eval[0],data_eval[1],data_eval[2], last1, last2,last3]):
            i = entry['idx']
            fig_img, ax_img = plt.subplots(1, 1, figsize=(1, 1))
            fig_pred, ax_pred = plt.subplots(1, 1, figsize=(1, 1))
            fig_label, ax_label = plt.subplots(1, 1, figsize=(1, 1))

            if isinstance(dataset, DataLoaderSWED) or isinstance(dataset, DataLoaderSNOWED_NDWI):
                rgb = np.moveaxis(dataset[i][0].numpy(), 0, -1)[:,:,[0,1,2]]
                real = np.moveaxis(dataset[i][1].numpy(), 0, -1)
            else:
                rgb = dataset[i][0][0][:,:,:,[0,1,2]][0]

            rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
            percentile = np.percentile(rgb, 99)
            clipped = np.clip(rgb, 0, percentile) / percentile

            with torch.no_grad():
                outputs = m(dataset[i][0].to(device).unsqueeze(0))
                if isinstance(outputs, tuple) and len(outputs):
                    predicted = softmax(outputs[0])
                else:
                    predicted = softmax(outputs)
                
                if no_predictions:
                    ax_img.imshow(clipped)
                    ax_label.imshow(np.argmax(real,axis=-1))

                    ax_img.set_xticks([])
                    ax_img.set_yticks([])

                    ax_label.set_xticks([])
                    ax_label.set_yticks([])
                else:
                    ax_img.imshow(clipped)
                    ax_pred.imshow(np.argmax(predicted.cpu().numpy()[0],axis=0))
                    ax_label.imshow(np.argmax(real,axis=-1))

                    ax_img.set_xticks([])
                    ax_img.set_yticks([])

                    ax_pred.set_xticks([])
                    ax_pred.set_yticks([])

                    ax_label.set_xticks([])
                    ax_label.set_yticks([])

            if no_predictions:
                fig_img.subplots_adjust(0, 0, 1, 1, 0, 0)
                fig_label.subplots_adjust(0, 0, 1, 1, 0, 0)
                if save:
                    # fig.set_size_inches(2.4,4.75)
                    fig_img.savefig(f'{results_folder}/img_{row_num}_{results_file}_no_pred.jpg', dpi=600)
                    fig_label.savefig(f'{results_folder}/label_{row_num}_{results_file}_no_pred.jpg', dpi=600)
                if display:
                    fig_img.show()
                    fig_label.show()
                plt.close(fig_img)
                plt.close(fig_pred)
                plt.close(fig_label)

            else:
                fig_img.subplots_adjust(0, 0, 1, 1, 0, 0)
                fig_pred.subplots_adjust(0, 0, 1, 1, 0, 0)
                fig_label.subplots_adjust(0, 0, 1, 1, 0, 0)
                if save:
                    # fig.set_size_inches(2.4,4.75)
                    fig_img.savefig(f'{results_folder}/img_{row_num}_{results_file}_{np.round(entry["iou"], 4)*100:.2f}_{np.round(entry["iou_water"], 4)*100:.2f}_{np.round(entry["iou_land"], 4)*100:.2f}.jpg', dpi=600)
                    fig_pred.savefig(f'{results_folder}/pred_{row_num}_{results_file}_{np.round(entry["iou"], 4)*100:.2f}_{np.round(entry["iou_water"], 4)*100:.2f}_{np.round(entry["iou_land"], 4)*100:.2f}.jpg', dpi=600)
                    fig_label.savefig(f'{results_folder}/label_{row_num}_{results_file}_{np.round(entry["iou"], 4)*100:.2f}_{np.round(entry["iou_water"], 4)*100:.2f}_{np.round(entry["iou_land"], 4)*100:.2f}.jpg', dpi=600)
                if display:
                    fig_img.show()
                    fig_pred.show()
                    fig_label.show()
                plt.close(fig_img)
                plt.close(fig_pred)
                plt.close(fig_label)
    else:
        number_of_rows = 4 if no_predictions else min(6, len(val_epoch_progress))
        number_of_cols = 2 if no_predictions else 3
        fig, ax = plt.subplots(number_of_rows, number_of_cols, figsize=(number_of_cols,number_of_rows))
        j = 0
        for entry in [data_eval[0],data_eval[1],data_eval[2], last1, last2,last3]:
            i = entry['idx']
            print(entry['iou'])
        # for i in range(len(predicted)):
            if isinstance(dataset, DataLoaderSWED) or isinstance(dataset, DataLoaderSNOWED_NDWI):
                rgb = np.moveaxis(dataset[i][0].numpy(), 0, -1)[:,:,[0,1,2]]
                real = np.moveaxis(dataset[i][1].numpy(), 0, -1)
            else:
                rgb = dataset[i][0][0][:,:,:,[0,1,2]][0]
            rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
            # clipped = np.clip(rgb, 0, 0.6)/0.6
            percentile = np.percentile(rgb, 99)
            clipped = np.clip(rgb, 0, percentile) / percentile
            # clipped = rgb
            with torch.no_grad():
                outputs = m(dataset[i][0].to(device).unsqueeze(0))
                if isinstance(outputs, tuple) and len(outputs):
                    predicted = softmax(outputs[0])
                else:
                    predicted = softmax(outputs)

            if no_predictions:
                ax[j, 0].imshow(clipped)
                ax[j, 1].imshow(np.argmax(real,axis=-1))

                ax[j, 0].set_xticks([])
                ax[j, 0].set_yticks([])

                ax[j, 1].set_xticks([])
                ax[j, 1].set_yticks([])
            elif number_of_rows > 1:

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
            if no_predictions and j == number_of_rows:
                plt.subplots_adjust(0, 0, 1, 1, 0, 0)
                if save:
                    # fig.set_size_inches(2.4,4.75)
                    plt.savefig(rf'{results_file}/{results_file}.jpg', dpi=600)
                if display:
                    plt.show()
                fig, ax = plt.subplots(5, 3)
                j = 0
            elif j == number_of_rows:
                # plt.subplots_adjust(0.021, 0.012, 0.376, 0.998, 0, 0.067)
                plt.subplots_adjust(0, 0, 1, 1, 0, 0)
                if save:
                    fig.set_size_inches(2.4,4.75)
                    plt.savefig(f'{results_file}/{results_file}.jpg', dpi=600)
                if display:
                    plt.show()
                fig, ax = plt.subplots(5, 3)
                j = 0
            print(j)

def test_one_folder(folder, opts, result_folder=None, config=None):
    BATCH_SIZE = 1
    weights = rf'.\{folder}\BEST.pt'
    device = opts['device']
    if opts['quant']:
        device = 'cpu'

    NDWI = config['NDWI'] if 'NDWI' in config else False
    SCALE = config['SCALE'] if 'SCALE' in config else 1

    if 'DeepUNet' in folder:
        channels = 4 if NDWI else 3
        output_count = config['output_count'] if 'output_count' in config else 1
        m = DeepUNet(channels, SCALE, outputs=output_count)
        m.to(device)
        m.load_state_dict(torch.load(weights))
    elif 'DeepUNet2' in folder:
        m = DeepUNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'SeNet2' in folder:
        m = SeNet2.get_model(data_loader_train.input_size, BATCH_SIZE)
    elif 'MU_Net' in folder:
        output_count = config['output_count'] if 'output_count' in config else 1
        upsample_mode = config['UPSAMPLE_MODE'] if 'UPSAMPLE_MODE' in config else 'bilinear'
        encoder_channels = [i // SCALE for i in [4,64,128,256,512]]
        encoder_channels[0] = 4 if NDWI else 3
        base_c = encoder_channels[1]
        ABLATION = config['ABLATION'] if 'ABLATION' in config else 0
        INCLUDE_AMM = config['INCLUDE_AMM'] if 'INCLUDE_AMM' in config else True
        
        # m = MU_Net([32,64,128,256], base_c=32, bilinear=False)
        m = MU_Net(encoder_channels, upsample_mode=upsample_mode, base_c = base_c, outputs=output_count, ablation=ABLATION, include_AMM=INCLUDE_AMM) #1st NDWI #OSIOSN1
        # m = MU_Net([4, 32,64,128,256], base_c=32) #OSIOSN2
        # m = experimental.MUNet([4, 32,64,128,256], base_c=32)

        # m = MU_Net()
        # m = experimental.MUNet(encoder_channels=[4,32,64,128,256], base_c = 16)
        m.to(device)
        m.load_state_dict(torch.load(weights))

        print(output_count)
        print(upsample_mode)
        print(encoder_channels)
        print(base_c)
        print(ABLATION)
        print(INCLUDE_AMM)
        print(f'Total params: {sum(p.numel() for p in m.parameters())}')
        print(f'Trainable params: {sum(p.numel() for p in m.parameters() if p.requires_grad)}')
    elif 'SeNet' in folder:
        channels = 4 if NDWI else 3
        output_count = config['output_count'] if 'output_count' in config else 2
        m = SeNet(channels, SCALE, outputs=output_count)
        m.to(device)
        m.load_state_dict(torch.load(weights))

    if 'SWED' in folder:
        PATH = rf'.\SWED\test'
        img_files, label_files = get_file_names(PATH, '.tif', 'SWED')
        if NDWI:
            data_set_test = DataLoaderSWED_NDWI(img_files, label_files, False, inference=False)
        else:
            data_set_test = DataLoaderSWED(img_files, label_files, False)
        

    elif 'SNOWED' in folder:
        PATH = rf'.\SNOWED_TEST'
        img_files = get_file_names(PATH, '.npy', 'SNOWED')
        # data_set_test = DataLoaderSNOWED(img_files, False)
        data_set_test = DataLoaderSNOWED_NDWI(img_files, False, root_path=PATH)

    elif 'SWED_FULL' in folder:
        PATH = rf'.\SWED\test'
        img_files, label_files = get_file_names(PATH, '.tif', 'SWED')
        data_set_test = DataLoaderSWED(img_files, label_files, False)

    loader = DataLoader(data_set_test, batch_size=BATCH_SIZE, shuffle=False)

    plot_settings = {
        'save': True,
        'display': False,
        'no_predictions': False,
        'separated': True
    }

    if plot_settings['save'] and result_folder is None:
        raise Exception('Results folder not specified')
    else:
        file = folder.split('\\')[-1]
        result_folder = rf"{result_folder}"
        results_file = rf"{file}"
        test(loader, data_set_test, m, plot_settings, result_folder, results_file, opts, device)


if __name__ == '__main__':
    base = rf'weights\MU_Net\REPORT 20-04-2024'

    for d in ['cuda']:
        optimize_options = {
            'quant': False,
            'prune': False,
            'bn fold': False,
            'quant dtype': torch.qint8,
            'prune amount': 0.3,
            'device': d
        }
    # QUANT = False
    # PRUNE = True
    # BN_FOLD = False
    # folder = rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample' #1)?
        # folder = rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample' #2)? #OSIOSN #1
    # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample' #3)?
    # folder = rf'weights\MU_Net\2024-04-16 12_50_21 Dice+Crossentropy SWED_FULL 1e-03 sample' #4)?
    # folder = rf'weights\MU_Net\2024-05-25 13_45_24 Dice+Crossentropy SWED 1e-03 sample'
    # folder = rf'weights\MU_Net\2024-05-30 15_16_40 Dice+Crossentropy SWED_FULL 1e-03 sample'

    #1st NDWI
    # folder = rf'weights\MU_Net\2024-04-14 17_55_05 Dice+Crossentropy SWED 1e-03 sample'

    # BEST FULL SWED
        # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample' #OSIOSN #2 smaller

    # folder = rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample' #SGD

    # folder = rf'weights\MU_Net\2024-05-27 01_38_49 Dice+Crossentropy SWED_FULL 1e-03 sample'

    # folder = rf'weights\MU_Net\2024-05-28 03_00_37 Dice+Crossentropy SWED_FULL 1e-03 sample'

    # no split
    # folder = rf'weights\MU_Net\2024-05-29 01_55_09 Dice+Crossentropy SWED_FULL 1e-03 sample'

    #OG
    # folder = rf'weights\MU_Net\2024-09-08 18_16_56 Dice+Crossentropy SWED_FULL 1e-03 sample'

    #SGD
        # folder = rf'weights\MU_Net\2024-09-10 03_21_11 Dice+Crossentropy SWED_FULL 1e-03 sample'
    
    #tab52 \tab52\#{i}
        # folders = {
        #     1: rf'weights\DeepUNet\24_12_2023 Sorensen_Dice SNOWED.jpg',
        #     2: rf'weights\DeepUNet\2023-12-29 04_53_30 Weighted_Dice SNOWED 1e-06 sample.jpg',
        #     3: rf'weights\DeepUNet\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample.jpg',
        #     4: rf'weights\DeepUNet\26_12_2023 Sorensen_Dice SWED sample.jpg',
        #     5: rf'weights\DeepUNet\26_12_2023 Weighted_Dice SWED sample.jpg',
        #     6: rf'weights\DeepUNet\26_12_2023 Weighted_Dice+Crossentropy SWED sample.jpg',
        #     7: rf'weights\DeepUNet\2023-12-29 03_33_39 Weighted_Dice+Crossentropy SWED 1e-06 sample.txt',
        #     8: rf'weights\SeNet\27_12_2023 Sorensen_Dice SNOWED sample.jpg',
        #     9: rf'weights\SeNet\27_12_2023 Weighted_Dice SNOWED sample.jpg',
        #     10: rf'weights\SeNet\26_12_2023 Weighted_Dice+Crossentropy SNOWED sample.jpg',
        #     11: rf'weights\SeNet\27_12_2023 Sorensen_Dice SWED sample.jpg',
        #     12: rf'weights\SeNet\27_12_2023 Weighted_Dice SWED sample.jpg',
        #     13: rf'weights\SeNet\27_12_2023 Weighted_Dice+Crossentropy SWED sample.jpg',
        #     14: rf'weights\SeNet\2023-12-29 02_11_31 Weighted_Dice+Crossentropy SWED 1e-06 sample.jpg',
        # }
    
    #NDWI \MU_NET_NDWI\#{i}
        # folders = {
            # 1: rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample',
            # 2: rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample',
            # 3: rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample',
            # 4: rf'weights\MU_Net\2024-04-16 12_50_21 Dice+Crossentropy SWED_FULL 1e-03 sample'
        # }
    #1.
    # folder rf'weights\MU_Net\2024-04-12 01_00_48 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #2.
    # folder = rf'weights\MU_Net\2024-04-14 20_18_09 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #3.
    # folder = rf'weights\MU_Net\2024-04-16 02_28_26 Dice+Crossentropy SWED_FULL 1e-03 sample'
    #4.
    # folder = rf'weights\MU_Net\2024-04-16 12_50_21 Dice+Crossentropy SWED_FULL 1e-03 sample'
        # SeNetLossvsDiceCE#2\#{i} SWED
        # folders = {
        #     1: rf'weights\SeNet\2024-10-03 22_17_06 SeNetLoss SWED_FULL 1e-03 sample',
        #     2: rf'weights\SeNet\2024-10-04 09_53_56 SeNetLoss SWED_FULL 1e-03 sample',
        #     3: rf'weights\SeNet\2024-10-05 12_38_26 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     4: rf'weights\SeNet\2024-10-05 16_04_39 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     5: rf'weights\DeepUNet\2024-10-04 14_44_12 SeNetLoss SWED_FULL 1e-03 sample',
        #     6: rf'weights\DeepUNet\2024-10-04 18_23_22 SeNetLoss SWED_FULL 1e-03 sample',
        #     7: rf'weights\DeepUNet\2024-10-05 01_40_26 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     8: rf'weights\DeepUNet\2024-10-05 05_28_33 Dice+Crossentropy SWED_FULL 1e-03 sample',
        # }

            # SeNetLossvsDiceCE#2\#{i} SNOWED
        # folders = {
        #     11: rf'weights\SeNet\2024-10-07 16_21_10 SeNetLoss SNOWED 1e-03 sample',
        #     12: rf'weights\SeNet\2024-10-07 16_34_49 SeNetLoss SNOWED 1e-03 sample',
        #     13: rf'weights\SeNet\2024-10-07 14_29_49 Dice+Crossentropy SNOWED 1e-03 sample',
        #     14: rf'weights\SeNet\2024-10-07 15_25_21 Dice+Crossentropy SNOWED 1e-03 sample',
        #     15: rf'weights\DeepUNet\2024-10-08 02_52_49 SeNetLoss SNOWED 1e-03 sample',
        #     16: rf'weights\DeepUNet\2024-10-08 03_39_53 SeNetLoss SNOWED 1e-03 sample',
        #     17: rf'weights\DeepUNet\2024-10-08 01_00_44 Dice+Crossentropy SNOWED 1e-03 sample',
        #     18: rf'weights\DeepUNet\2024-10-08 01_56_21 Dice+Crossentropy SNOWED 1e-03 sample',
        # }
            #MU_Net ablacje \MU_NET_ablacje\#{i}
        # folders = {
        #     2: rf'weights\MU_Net\2024-10-05 22_31_15 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     3: rf'weights\MU_Net\2024-10-06 04_18_33 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     4: rf'weights\MU_Net\2024-10-06 11_23_08 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     5: rf'weights\MU_Net\2024-10-06 18_30_12 Dice+Crossentropy SWED_FULL 1e-03 sample',
        #     6: rf'weights\MU_Net\2024-10-07 03_51_53 Dice+Crossentropy SWED_FULL 1e-03 sample',
        # }

            #MU_Net SeNetLoss SeNetLossvsDiceCE#2\#{i}
        folders = {
            # 9: rf'weights\MU_Net\2024-10-09 02_25_18 SeNetLoss SWED_FULL 1e-03 sample',
            # 10: rf'weights\MU_Net\2024-10-09 13_47_31 SeNetLoss SWED_FULL 1e-03 sample',
            # 19: rf'weights\MU_Net\2024-10-09 10_24_10 SeNetLoss SNOWED 1e-03 sample',
            20: rf'weights\MU_Net\2024-10-09 11_24_58 SeNetLoss SNOWED 1e-03 sample',
        }

        for i, folder in folders.items():
            res = rf'plots\NEW_IOU\new_contrast\MU_NET_SeNetLoss#2TESTIOU\#{i}'
            print(folder)
            with open(rf'{folder}\config.txt') as f:
                config = ast.literal_eval(f.read())
            os.makedirs(res, exist_ok=True)
            test_one_folder(folder, optimize_options, res, config)


        # for folder in os.listdir(base):
        #     test_one_folder(rf'{base}\{folder}', res)
        