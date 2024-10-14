from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QPlainTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QMainWindow, QFileDialog, QComboBox, QGridLayout, QSpinBox, QAbstractSpinBox
from PyQt5.QtCore import QProcess
from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt import FigureCanvasQT as FigureCanvas
# from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pytorch_generator import DataLoaderSWED_NDWI
from torch.utils.data import DataLoader

import ast
import numpy as np
import os
import sys

from gui.model_config_window import Model_Config_Window


class Train_Window(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Training")
        self.resize(1280, 720)

        wid = QWidget()
        self.setCentralWidget(wid)
        layout = QVBoxLayout()
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        self.config = {
            'MODEL': '<Not selected>',
            'EPOCHS': 100,
            'BATCH_SIZE': 10,
            'LEARNING_RATE': 1e-3,
            'TRAIN_PART': 0.7,
            'SCHEDULER': '<Not selected>',
            'DEBUG': '<Not selected>',
            'DATASET': '<Not selected>',
            'LOSS': '<Not selected>',
            'OPTIMIZER': '<Not selected>',
            'MOMENTUM': 0.9,
            'WEIGHT_DECAY': 1e-4,
            'PRECISION': 32,
            'NDWI': '<Not selected>',
            'EVAL_FREQ': 20,
            'SCALE': '<Not selected>',
            'ABLATION': '<Not selected>',
            'INCLUDE_AMM': '<Not selected>'
        }
        self.models = {
            'SeNet': 'SeNet class',
            'DeepUNet': 'DeepUNet class',
            'MU-Net': 'MU-Net class'
        }
        self.model_config_window = Model_Config_Window(self, True)

        self.cfg_button = QPushButton('Configure training')
        self.cfg_button.clicked.connect(self.show_config_window)

        self.train_button = QPushButton('Start training')
        self.train_button.clicked.connect(self.train)

        self.stop_button = QPushButton('Stop training')
        self.stop_button.clicked.connect(self.stop_train)
        self.stop_button.setEnabled(False)

        layout.addWidget(self.text)
        layout.addWidget(self.cfg_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.stop_button)
        wid.setLayout(layout)

    def closeEvent(self, event):
        # return super().closeEvent(a0)
        self.parent().show_main_window()
        event.accept()

    def show_config_window(self):
        if self.model_config_window.isHidden():
            self.model_config_window.show()
            self.change_model(self.model_config_window.model_cbox.currentText())
            self.change_optimizer(self.model_config_window.optimizer_cbox.currentText())

    def message(self, s):
        self.text.appendPlainText(s)

    def training_ended(self):
        self.message("Training finished.")
        self.process = None

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        if state_name == 'Running':
            self.train_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        elif state_name == 'Not running':
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        self.message(f"State changed: {state_name}")

    def train(self):
        if self.validate_config(self.config):
            self.message("Starting training...")
            self.process = QProcess()
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.stateChanged.connect(self.handle_state)
            args = ['execute_training.py']
            # args = ['train.py']
            config = self.config_to_args()
            args.extend(config)
            path_to_venv_python = sys.executable
            self.process.start(path_to_venv_python, args)
            self.process.finished.connect(self.training_ended)

    def stop_train(self):
        self.process.kill()
        self.stop_button.setEnabled(False)
        self.train_button.setEnabled(True)

    def config_to_args(self):
        args = []
        for key, value in self.config.items():
            args.append(f'--{key}')
            args.append(f'{value}'.replace('-','_'))

        return args

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
                    elif value == '<Not selected>' and key == 'ABLATION':
                        self.config['ABLATION'] = 0
                    elif value == '<Not selected>' and key == 'INCLUDE_AMM':
                        self.config['INCLUDE_AMM'] = True
        else:
            filled = False

        return filled

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

    def change_epochs(self, v):
        self.config['EPOCHS'] = int(v) if v != '<Not selected>' else v
        self.model_config_window.eval_freq_sbox.setRange(0, v)

    def change_batchsize(self, v):
        self.config['BATCH_SIZE'] = int(v) if v != '<Not selected>' else v

    def change_lrate(self, v):
        self.config['LEARNING_RATE'] = float(v) if v != '<Not selected>' else v

    def change_train_part(self, v):
        self.config['TRAIN_PART'] = float(v) if v != '<Not selected>' else v

    def change_scheduler(self, v):
        self.config['SCHEDULER'] = v

    def change_debug(self, v):
        self.config['DEBUG'] = not ast.literal_eval(v) if v != '<Not selected>' else v

    def change_dataset(self, v):
        if v == 'SWED FULL':
            self.config['DATASET'] = 'SWED_FULL'
        elif v == 'SWED sample':
            self.config['DATASET'] = 'SWED'
        else:
            self.config['DATASET'] = v

    def change_loss(self, v):
        self.config['LOSS'] = v

    def change_optimizer(self, v):
        self.config['OPTIMIZER'] = v
        # if v == 'SGD':
        #     self.model_config_window.momentum_sbox.setEnabled(True)
        #     self.model_config_window.weight_decay_sbox.setEnabled(True)
        # else:
        #     self.model_config_window.momentum_sbox.setEnabled(False)
        #     self.model_config_window.weight_decay_sbox.setEnabled(False)

    def change_momentum(self, v):
        self.config['MOMENTUM'] = float(v) if v != '<Not selected>' else v

    def change_weight_decay(self, v):
        self.config['WEIGHT_DECAY'] = float(v) if v != '<Not selected>' else v

    def change_eval_freq(self, v):
        self.config['EVAL_FREQ'] = int(v) if v != '<Not selected>' else v

    def change_model(self, name):
        if name in self.models:
            # self.model = self.models[name]#()
            self.config['MODEL'] = name
            if name == 'MU-Net':
                self.model_config_window.ablation_cbox.setEnabled(True)
                self.model_config_window.amm_cbox.setEnabled(True)
            else:
                self.model_config_window.ablation_cbox.setEnabled(False)
                self.model_config_window.amm_cbox.setEnabled(False)
            print(self.config['MODEL'])
        else:
            print('Invalid model name')
            self.model_config_window.ablation_cbox.setEnabled(False)
            self.model_config_window.amm_cbox.setEnabled(False)