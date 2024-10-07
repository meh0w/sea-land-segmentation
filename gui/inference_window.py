from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QMainWindow, QFileDialog, QComboBox, QGridLayout
from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pytorch_generator import DataLoaderSWED_NDWI
from torch.utils.data import DataLoader

import ast
import numpy as np
import os

from model_config_window import Model_Config_Window


class Inference_Window(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Inference")
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

        self.cfg_button = QPushButton('Setup config manually')
        self.cfg_button.clicked.connect(self.show_config_window)

        self.display_button = QPushButton('Display')
        self.display_button.clicked.connect(self.display_images)

        self.model = None

        self.models = {
            'SeNet': 'SeNet class',
            'DeepUNet': 'DeepUNet class',
            'MU-Net': 'MU-Net class'
        }

        self.config = {
            'MODEL': '<Not selected>',
            'PRECISION': '<Not selected>',
            'NDWI': '<Not selected>',
            'SCALE': '<Not selected>',
            'ABLATION': '<Not selected>',
            'INCLUDE_AMM': '<Not selected>'
        }
        self.image_files = []
        self.label_files = []

        self.model_config_window = Model_Config_Window(self, False)

        self.data_set = None

        wid = QWidget()
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        options = QWidget()
        options_layout = QGridLayout()
        options_layout.addWidget(self.img_file_button, 0, 0)
        options_layout.addWidget(self.mask_file_button, 0, 1)
        options_layout.addWidget(self.cfg_file_button, 1, 0)
        options_layout.addWidget(self.cfg_button, 1, 1)
        options.setLayout(options_layout)
        layout.addWidget(options)
        layout.addWidget(self.display_button)

        wid.setLayout(layout)

    def closeEvent(self, event):
        # return super().closeEvent(a0)
        self.parent().show_main_window()
        event.accept()

    def show_config_window(self):
        if self.model_config_window.isHidden():
            self.model_config_window.show()
            self.change_model(self.model_config_window.model_cbox.currentText())

    def plot(self):

        self.figure.clear()
        n_rows = len(self.data_set)
        print(f'{n_rows=}')
        n_cols = 3

        img_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + 1) for idx in range(n_rows)]
        pred_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + 2) for idx in range(n_rows)]
        label_axs = [self.figure.add_subplot(n_rows, n_cols, (idx*n_cols) + 3) for idx in range(n_rows)]
        # img_ax.clear()
        # label_ax.clear()
        # self.figure.clear()

        for i in range(n_rows):
            img_axs[i].clear()
            pred_axs[i].clear()
            label_axs[i].clear()

            img = self.data_set[i][0]
            if img is not None:
                rgb = np.moveaxis(img.numpy(), 0, -1)[:,:,[0,1,2]]
                rgb = (rgb-np.min(rgb))/(np.max(rgb)-np.min(rgb))
                clipped = np.clip(rgb, 0, 0.6) / 0.6
                img_axs[i].imshow(clipped)

                if self.model is not None:
                    #img = img.to(device)
                    #outputs = self.model(img)
                    pred_axs[i].imshow(clipped)

            label = self.data_set[i][1]
            if label is not None:
                real = np.moveaxis(label.numpy(), 0, -1)[:,:,[0,1,2]]
                label_axs[i].imshow(clipped)
            
            self.remove_ticks(img_axs[i])
            self.remove_ticks(pred_axs[i])
            self.remove_ticks(label_axs[i])

        self.canvas.draw()

    def remove_ticks(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
    
    def get_img_files(self):
        image_files = QFileDialog.getOpenFileNames(self, 'Open file', 'c:\\',"Image files (*.tif *.tiff)")[0]
        if len(image_files) > 0:
            self.image_files = sorted(image_files)

    
    def get_mask_files(self):
        mask_files = QFileDialog.getOpenFileNames(self, 'Open file', 'c:\\',"Image files (*.tif *.tiff)")[0]
        if len(mask_files) > 0:
            self.label_files = sorted(mask_files)


    def display_images(self):
        self.data_set = DataLoaderSWED_NDWI(self.image_files, self.label_files)
        self.loader = DataLoader(self.data_set, batch_size=1, shuffle=False)
        self.plot()

    def change_model(self, name):
        if name in self.models:
            self.model = self.models[name]#()
            if name == 'MU-Net':
                self.model_config_window.ablation_cbox.setEnabled(True)
                self.model_config_window.amm_cbox.setEnabled(True)
            else:
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
                except:
                    print('Invalid config file')

            filled = self.validate_config(config)

    def validate_config(self, config):
        filled = True
        if type(config) == dict:
            if config['MODEL'] == self.model_config_window.model_cbox.currentText():
                self.config['MODEL'] = config['MODEL']

            if config['PRECISION'] in [16, 32]:
                self.config['PRECISION'] = config['PRECISION']

            if config['NDWI'] in [True, False]:
                self.config['NDWI'] = config['NDWI']

            if config['SCALE'] in [1, 2]:
                self.config['SCALE'] = config['SCALE']

            if config['ABLATION'] in [0, 1, 2, 3, 4]:
                self.config['ABLATION'] = config['ABLATION']

            if config['INCLUDE_AMM'] in [True, False]:
                self.config['INCLUDE_AMM'] = config['INCLUDE_AMM']

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
    
    def change_precision(self, v):
        self.config['PRECISION'] = v

    def change_NDWI(self, v):
        self.config['NDWI'] = v

    def change_scale(self, v):
        self.config['SCALE'] = v

    def change_ablation(self, v):
        self.config['ABLATION'] = v

    def change_amm(self, v):
        self.config['INCLUDE_AMM'] = v