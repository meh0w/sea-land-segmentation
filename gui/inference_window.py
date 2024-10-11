from PyQt5.QtGui import QFontMetrics, QCloseEvent
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPlainTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QMainWindow, QFileDialog, QComboBox, QGridLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as mpatches
from pytorch_generator import DataLoader_inference
from pytorch_metrics_new import All_metrics
from torch.utils.data import DataLoader
import torch

import ast
import numpy as np

from gui.model_config_window import Model_Config_Window
from pytorch_DeepUNet import DeepUNet
from pytorch_SeNet import SeNet
from pytorch_MU_NET import MUNet as MU_Net
import os


class Inference_Window(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Inference")
        self.resize(1280, 720)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.button = QPushButton('plot')
        self.button.move(50, 100)
        self.button.clicked.connect(self.plot)
        
        self.img_file_button = QPushButton('Choose images')
        self.img_file_button.clicked.connect(self.get_img_files)

        self.mask_file_button = QPushButton('Choose ground truth masks')
        self.mask_file_button.clicked.connect(self.get_mask_files)

        self.cfg_file_button = QPushButton('Choose model config file')
        self.cfg_file_button.clicked.connect(self.get_config)

        self.weights_file_button = QPushButton('Choose model weights file')
        self.weights_file_button.clicked.connect(self.get_weights)

        self.cfg_button = QPushButton('Setup config manually')
        self.cfg_button.clicked.connect(self.show_config_window)

        self.cfg_reset_button = QPushButton('Reset model config')
        self.cfg_reset_button.clicked.connect(self.reset_config)

        self.weight_reset_button = QPushButton('Reset model weights')
        self.weight_reset_button.clicked.connect(self.reset_weights)

        self.display_button = QPushButton('Display')
        self.display_button.clicked.connect(self.display_images)

        # self.predict_button = QPushButton('Predict')
        # self.predict_button.clicked.connect(self.predict)

        self.model = None
        self.weights_file = None

        self.models = {
            'SeNet': SeNet,
            'DeepUNet': DeepUNet,
            'MU-Net': MU_Net,
            'MU_Net': MU_Net
        }

        self.config = {
            'MODEL': '<Not selected>',
            'PRECISION': '<Not selected>',
            'NDWI': '<Not selected>',
            'SCALE': '<Not selected>',
            'ABLATION': '<Not selected>',
            'INCLUDE_AMM': '<Not selected>',
            'DATA_LOADER': '<Not selected>',
            'OUTPUTS': '<Not selected>',
        }
        self.image_files = []
        self.label_files = []

        self.model_config_window = Model_Config_Window(self, False)

        self.data_set = None

        wid = QWidget()
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        
        font = self.text_box.document().defaultFont()
        fontMetrics = QFontMetrics(font)
        textSize = fontMetrics.size(0, self.text_box.toPlainText())
        h = textSize.height() + 10
        self.text_box.setFixedHeight(h*4)
        layout.addWidget(self.text_box)


        options = QWidget()
        options_layout = QGridLayout()
        options_layout.addWidget(self.img_file_button, 0, 0)
        options_layout.addWidget(self.mask_file_button, 0, 1)
        options_layout.addWidget(self.cfg_file_button, 1, 0)
        options_layout.addWidget(self.cfg_button, 1, 1)

        options_layout.addWidget(self.cfg_reset_button, 2, 0)
        options_layout.addWidget(self.weight_reset_button, 2, 1)

        options_layout.setRowStretch(2, 1)
        options.setLayout(options_layout)
        
        layout.addWidget(options)

        options2 = QWidget()
        options2_layout = QGridLayout()
        options2_layout.addWidget(self.weights_file_button, 1, 0)
        # layout.addWidget(self.predict_button)
        options2_layout.addWidget(self.display_button,2 ,0)
        options2.setLayout(options2_layout)
        layout.addWidget(options2)

        layout.setAlignment(Qt.AlignBottom)
        # layout.setRowStretch(3, 1)
        wid.setLayout(layout)

    def closeEvent(self, event):
        # return super().closeEvent(a0)
        self.parent().show_main_window()
        event.accept()

    def show_config_window(self):
        if self.model_config_window.isHidden():
            self.model_config_window.show()
            self.model_config_window.refresh_values()
            # self.change_model(self.model_config_window.model_cbox.currentText())
    
    def message(self, s):
        self.text_box.appendPlainText(s)

    def plot(self):

        self.figure.clear()
        n_rows = len(self.data_set)
        print(f'{n_rows=}')
        n_cols = 0

        col_img = 1
        col_pred = 2
        col_label = 3

        indexes = {
            1: 'a',
            2: 'b',
            3: 'c'
        }

        legend_handles = []
        first_img = self.data_set[0][0]
        first_label = self.data_set[0][1]

        if len(self.data_set) > 0:
            if self.validate_config(self.config):
                self.load_model()
            if first_img is not None:
                n_cols += 1
            else:
                col_img = 0
                col_pred -= 1
                col_label -= 1
            if self.model is not None:
                n_cols += 1
            else:
                col_pred = 0
                col_label -= 1
            if first_label is not None:
                n_cols += 1
            else:
                col_label = 0

        print(f'{n_cols=}, {col_img=}, {col_pred=}, {col_label=}')

        img_axs, pred_axs, label_axs = None, None, None
        if col_img > 0:
            img_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + col_img) for idx in range(n_rows)]
        if col_pred > 0:
            pred_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + col_pred) for idx in range(n_rows)]
        if col_label > 0:
            label_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + col_label) for idx in range(n_rows)]

        metrics = All_metrics(self.device, '[TEST]')
        for i in range(n_rows):
            outputs_plotted = False
            label_plotted = False
            
            try:
                img, img_error = self.data_set[i][0], self.data_set[i][2][0]
                if img is not None:
                        img_axs[i].clear()
                        rgb = np.moveaxis(img.numpy(), 0, -1)[:,:,[0,1,2]]
                        rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
                        percentile = np.percentile(rgb, 99)
                        clipped = np.clip(rgb, 0, percentile) / percentile
                        img_axs[i].imshow(clipped.astype(np.float32))
                        if i == 0:
                            img_axs[i].set_title(f'{indexes[col_img]})')
                            img_handle = mpatches.Patch(color='none', label=f'{indexes[col_img]}) Input images')
                            legend_handles.append(img_handle) 
                        self.remove_ticks(img_axs[i])
                else:
                    if img_axs is not None:
                        img_axs[i].set_visible(False)
                if img_error:
                    self.message("ERROR: Something went wrong with displaying input images!")
                    if img_axs is not None:
                        img_axs[i].set_visible(False)
                    
            except:
                self.message("ERROR: Something went wrong with displaying input images!")
                if img_axs is not None:
                    img_axs[i].set_visible(False)

            # self.load_model()
            if self.model is not None:
                if self.data_set[i][0] is not None:
                    try:
                        pred_axs[i].clear()
                        with torch.no_grad():
                            self.model.eval()
                            self.model.to(self.data_set.precision)
                            print(self.data_set.precision)
                            softmax = torch.nn.Softmax(dim=1)
                            img = img.to(self.device)
                            outputs = self.model(self.data_set[i][0].to(self.device).unsqueeze(0))
                            if isinstance(outputs, tuple) and len(outputs) == 2:
                                predicted = softmax(outputs[0])
                            else:
                                predicted = softmax(outputs)
                            pred_axs[i].imshow(np.argmax(predicted.cpu().numpy()[0],axis=0))
                            outputs_plotted = True
                            if i == 0:
                                pred_axs[i].set_title(f'{indexes[col_pred]})')
                                pred_handle = mpatches.Patch(color='none', label=f'{indexes[col_pred]}) Segmentation results')
                                legend_handles.append(pred_handle)
                            self.remove_ticks(pred_axs[i])
                    except Exception as e:
                        self.message(f"ERROR: Something went wrong with displaying segmentation results! \n {e}")
                        if pred_axs is not None:
                            pred_axs[i].set_visible(False)
                elif self.data_set[i][2][0]:
                    self.message("ERROR: Something went wrong with displaying segmentation results!")
                    if pred_axs is not None:
                        pred_axs[i].set_visible(False)
                else:
                    if pred_axs is not None:
                        pred_axs[i].set_visible(False)
            else:
                if pred_axs is not None:
                    pred_axs[i].set_visible(False)
                self.message("ERROR: Model is not configured")

            try:
                label, label_error = self.data_set[i][1], self.data_set[i][2][1]
                if label is not None:
                    label_axs[i].clear()
                    real = np.moveaxis(label.numpy(), 0, -1)
                    label_axs[i].imshow(np.argmax(real,axis=-1))
                    label_plotted = True
                    if i == 0:
                        label_axs[i].set_title(f'{indexes[col_label]})')
                        label_handle = mpatches.Patch(color='none', label=f'{indexes[col_label]}) Ground truth masks')
                        legend_handles.append(label_handle)
                    self.remove_ticks(label_axs[i])
                else:
                    if label_axs is not None:
                        label_axs[i].set_visible(False)
                if label_error:
                    if label_axs is not None:
                        label_axs[i].set_visible(False)
                    self.message("ERROR: Something went wrong with displaying ground truth masks!")
            except:
                if label_axs is not None:
                    label_axs[i].set_visible(False)
                self.message("ERROR: Something went wrong with displaying ground truth masks!")

            if outputs_plotted and label_plotted:
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    out = outputs[0]
                else:
                    out = outputs
                metrics.calc(out.to(self.device), label.unsqueeze(0).to(self.device))
                results = metrics.get(False)
                pred_axs[i].set_xlabel(f'IoU = {results[f"{metrics.prefix} IoU mean"][i].item()*100:.2f}%')
        
        self.figure.subplots_adjust(
            top=0.925,
            bottom=0.105,
            left=0.012,
            right=0.988,
            hspace=0.291,
            wspace=0.0
            )
        self.figure.tight_layout()
        self.figure.legend(handles=legend_handles, loc="lower left", ncol=n_cols)
        self.canvas.draw()

    def remove_ticks(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
    
    def get_img_files(self):
        image_files = QFileDialog.getOpenFileNames(self, 'Open file', 'c:\\',"Image files (*.tif *.tiff)")[0]
        if len(image_files) > 0:
            self.image_files = sorted(image_files)
            self.message(f'Selected input images: \n')
            for file in self.image_files:
                self.message(f'{file}\n')

    
    def get_mask_files(self):
        mask_files = QFileDialog.getOpenFileNames(self, 'Open file', 'c:\\',"Image files (*.tif *.tiff)")[0]
        if len(mask_files) > 0:
            self.label_files = sorted(mask_files)
            self.message(f'Selected ground truth masks: \n')
            for file in self.label_files:
                self.message(f'{file}\n')
            


    def display_images(self):
        if max(len(self.image_files), len(self.label_files)) > 0 and self.validate_dataset():
            self.data_set = DataLoader_inference(self.image_files, self.label_files, self.config['PRECISION'], self.config['NDWI'], self.config['DATA_LOADER'])
            self.loader = DataLoader(self.data_set, batch_size=1, shuffle=False)
            self.plot()
        elif not (max(len(self.image_files), len(self.label_files)) > 0):
                self.message('ERROR: No files to display have been provided.')
        elif not (self.validate_dataset()):
            self.message('ERROR: Missing atleast one of the following settings: [PRECISION, NDWI, DATA LOADER].')

    def load_model(self):
        if self.config['MODEL'] == 'SeNet':
            channels = 4 if self.config['NDWI'] else 3
            output_count = self.config['OUTPUTS'] if 'OUTPUTS' in self.config else 2
            self.model = SeNet(channels, self.config['SCALE'], outputs=output_count)
            self.model.to(self.device)
        if self.config['MODEL'] == 'DeepUNet':
            channels = 4 if self.config['NDWI'] else 3
            output_count = self.config['OUTPUTS'] if 'OUTPUTS' in self.config else 1
            self.model = DeepUNet(channels, self.config['SCALE'], outputs=output_count)
            self.model.to(self.device)
        if self.config['MODEL'] == 'MU-Net' or self.config['MODEL'] == 'MU_Net':
            encoder_channels = [i // self.config['SCALE'] for i in [4,64,128,256,512]]
            encoder_channels[0] = 4 if self.config['NDWI'] else 3
            output_count = self.config['OUTPUTS'] if 'OUTPUTS' in self.config else 1
            self.model = MU_Net(encoder_channels=encoder_channels, base_c = encoder_channels[1], outputs=output_count, ablation=self.config['ABLATION'], include_AMM=self.config['INCLUDE_AMM'])
            self.model.to(self.device)

        if self.model is not None and self.weights_file is not None:
            try:
                self.model.load_state_dict(torch.load(self.weights_file))
                # print(torch.load(self.weights_file))
            except Exception as e:
                self.message(f'ERROR: Something went wrong with loading model weights! \n {e}')

    def change_model(self, name):
        if name in self.models:
            if name == 'MU-Net' or name == 'MU_Net':
                self.config['MODEL'] = 'MU-Net'
                self.model_config_window.ablation_cbox.setEnabled(True)
                self.model_config_window.amm_cbox.setEnabled(True)
            else:
                self.config['MODEL'] = name
                self.model_config_window.ablation_cbox.setEnabled(False)
                self.model_config_window.amm_cbox.setEnabled(False)
            print(self.model)
        else:
            print('Invalid model name')
            self.model_config_window.ablation_cbox.setEnabled(False)
            self.model_config_window.amm_cbox.setEnabled(False)

    def get_config(self):
        config_file = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Text files (*.txt)")[0]
        if config_file.endswith('.txt') and os.path.isfile(config_file):
            with open(config_file, 'r') as f:
                try:
                    config = ast.literal_eval(f.read())
                    self.message(f'Selected model config: \n {config_file}')
                except:
                    print('Invalid config file')

            filled = self.validate_config(config)

    def get_weights(self):
        weights_file = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Weights files (*.pt)")[0]
        if weights_file.endswith('.pt') and os.path.isfile(weights_file):
            self.weights_file = weights_file
            self.message(f'Selected weights file: \n {self.weights_file}')

    def validate_config(self, config):
        filled = True
        if type(config) == dict:
            if 'MODEL' in config and config['MODEL'] in self.models:
                if config['MODEL'] == 'MU-Net' or config['MODEL'] == 'MU_Net':
                    self.config['MODEL'] = 'MU-Net'
                else:
                    self.config['MODEL'] = config['MODEL']

            if 'PRECISION' in config and config['PRECISION'] in [16, 32]:
                self.config['PRECISION'] = config['PRECISION']

            if 'NDWI' in config and config['NDWI'] in [True, False]:
                self.config['NDWI'] = config['NDWI']

            if 'SCALE' in config and config['SCALE'] in [1, 2]:
                self.config['SCALE'] = config['SCALE']

            if 'ABLATION' in config and config['ABLATION'] in [0, 1, 2, 3, 4]:
                self.config['ABLATION'] = config['ABLATION']

            if 'INCLUDE_AMM' in config and config['INCLUDE_AMM'] in [True, False]:
                self.config['INCLUDE_AMM'] = config['INCLUDE_AMM']

            if 'LOSS' in config and type(config['LOSS']) == str:
                if config['LOSS'].lower() == 'senetloss':
                    self.config['OUTPUTS'] = 2
                else:
                    self.config['OUTPUTS'] = 1

            if self.config['MODEL'] == 'MU-Net':
                for value in self.config.values():
                    if value == '<Not selected>':
                        filled = False
                        break

                if self.config['ABLATION']  == '<Not selected>' or self.config['INCLUDE_AMM']  == '<Not selected>':
                    filled = False
            else:
                for key, value in self.config.items():
                    if value == '<Not selected>' and key not in ['ABLATION', 'INCLUDE_AMM']:
                        filled = False
                        break
        else:
            filled = False

        return filled
    
    def validate_dataset(self):
        filled = True
        for key, value in self.config.items():
            if value == '<Not selected>' and key in ['PRECISION', 'NDWI', 'DATA_LOADER']:
                filled = False
                break
        return filled

    
    def reset_weights(self):
        self.weights_file = None

    def reset_config(self):
        self.config = {
            'MODEL': '<Not selected>',
            'PRECISION': '<Not selected>',
            'NDWI': '<Not selected>',
            'SCALE': '<Not selected>',
            'ABLATION': '<Not selected>',
            'INCLUDE_AMM': '<Not selected>',
            'DATA_LOADER': '<Not selected>',
            'OUTPUTS': '<Not selected>',
        }
    
    def change_precision(self, v):
        self.config['PRECISION'] = int(v) if v != '<Not selected>' else v

    def change_NDWI(self, v):
        self.config['NDWI'] = ast.literal_eval(v) if v != '<Not selected>' else v

    def change_scale(self, v):
        self.config['SCALE'] = int(v) if v != '<Not selected>' else v

    def change_ablation(self, v):
        self.config['ABLATION'] = int(v) if v != '<Not selected>' else 0

    def change_amm(self, v):
        self.config['INCLUDE_AMM'] = ast.literal_eval(v) if v != '<Not selected>' else True

    def change_loader(self, v):
        self.config['DATA_LOADER'] = v

    def change_outputs(self, v):
        self.config['OUTPUTS'] = int(v) if v != '<Not selected>' else v
