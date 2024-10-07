from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QMainWindow, QFileDialog, QComboBox, QGridLayout, QSpinBox, QAbstractSpinBox, QDoubleSpinBox
from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pytorch_generator import DataLoaderSWED_NDWI
from torch.utils.data import DataLoader

import ast
import numpy as np
import os

class Model_Config_Window(QMainWindow):
    def __init__(self, parent, training):
        super().__init__(parent)
        self.setWindowTitle("Choose model config")
        wid = QWidget()
        self.setCentralWidget(wid)
        layout = QGridLayout()

        self.model_cbox = QComboBox()
        self.model_cbox.addItems(['<Not selected>', 'SeNet', 'DeepUNet', 'MU-Net'])
        self.model_cbox.setCurrentText(self.parent().config['MODEL'])
        self.model_cbox.currentTextChanged.connect(self.parent().change_model)
        self.model_label = QLabel('Model: ')

        self.precision_cbox = QComboBox()
        self.precision_cbox.addItems(['<Not selected>', '16', '32'])
        self.precision_cbox.setCurrentText(self.parent().config['PRECISION'])
        self.precision_cbox.currentTextChanged.connect(self.parent().change_precision)
        self.precision_label = QLabel('Precision: ')

        self.NDWI_cbox = QComboBox()
        self.NDWI_cbox.addItems(['<Not selected>', 'True', 'False'])
        self.NDWI_cbox.setCurrentText(self.parent().config['NDWI'])
        self.NDWI_cbox.currentTextChanged.connect(self.parent().change_NDWI)
        self.NDWI_label = QLabel('NDWI: ')

        self.scale_cbox = QComboBox()
        self.scale_cbox.addItems(['<Not selected>', '1', '2'])
        self.scale_cbox.setCurrentText(self.parent().config['SCALE'])
        self.scale_cbox.currentTextChanged.connect(self.parent().change_scale)
        self.scale_label = QLabel('Scale: ')

        self.ablation_cbox = QComboBox()
        self.ablation_cbox.addItems(['<Not selected>', '0', '1', '2', '3', '4'])
        self.ablation_cbox.setCurrentText(self.parent().config['ABLATION'])
        self.ablation_cbox.currentTextChanged.connect(self.parent().change_ablation)
        self.ablation_label = QLabel('Ablation: ')

        self.amm_cbox = QComboBox()
        self.amm_cbox.addItems(['<Not selected>', 'True', 'False'])
        self.amm_cbox.setCurrentText(self.parent().config['INCLUDE_AMM'])
        self.amm_cbox.currentTextChanged.connect(self.parent().change_amm)
        self.amm_label = QLabel('Include AMM: ')

        layout.addWidget(self.model_cbox, 0, 1)
        layout.addWidget(self.precision_cbox, 1, 1)
        layout.addWidget(self.NDWI_cbox, 2, 1)
        layout.addWidget(self.scale_cbox, 3, 1)
        layout.addWidget(self.ablation_cbox, 4, 1)
        layout.addWidget(self.amm_cbox, 5, 1)

        layout.addWidget(self.model_label, 0, 0)
        layout.addWidget(self.precision_label, 1, 0)
        layout.addWidget(self.NDWI_label, 2, 0)
        layout.addWidget(self.scale_label, 3, 0)
        layout.addWidget(self.ablation_label, 4, 0)
        layout.addWidget(self.amm_label, 5, 0)

        if training:
            self.epochs_sbox = QSpinBox()
            self.epochs_sbox.setRange(0, 1000000)
            self.epochs_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.epochs_sbox.setValue(self.parent().config['EPOCHS'])
            self.epochs_sbox.valueChanged.connect(self.parent().change_epochs)
            self.epochs_label = QLabel('Number of epochs: ')

            self.batchsize_sbox = QSpinBox()
            self.batchsize_sbox.setRange(0, 1000)
            self.batchsize_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.batchsize_sbox.setValue(self.parent().config['BATCH_SIZE'])
            self.batchsize_sbox.valueChanged.connect(self.parent().change_batchsize)
            self.batchsize_label = QLabel('Batch size: ')

            self.lrate_sbox = QDoubleSpinBox()
            self.lrate_sbox.setRange(0, 10)
            self.lrate_sbox.setDecimals(5)
            self.lrate_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.lrate_sbox.setValue(self.parent().config['LEARNING_RATE'])
            self.lrate_sbox.valueChanged.connect(self.parent().change_lrate)
            self.lrate_label = QLabel('Learning rate: ')

            self.train_part_sbox = QDoubleSpinBox()
            self.train_part_sbox.setRange(0, 1)
            self.train_part_sbox.setDecimals(2)
            self.train_part_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.train_part_sbox.setValue(self.parent().config['TRAIN_PART'])
            self.train_part_sbox.valueChanged.connect(self.parent().change_train_part)
            self.train_part_label = QLabel('Training-Validation split: ')

            self.scheduler_cbox = QComboBox()
            self.scheduler_cbox.addItems(['<Not selected>', 'poly', 'custom', 'saturate'])
            self.scheduler_cbox.setCurrentText(self.parent().config['SCHEDULER'])
            self.scheduler_cbox.currentTextChanged.connect(self.parent().change_scheduler)
            self.scheduler_label = QLabel('Scheduler: ')

            self.debug_cbox = QComboBox()
            self.debug_cbox.addItems(['<Not selected>', 'True', 'False'])
            self.debug_cbox.setCurrentText(self.parent().config['DEBUG'])
            self.debug_cbox.currentTextChanged.connect(self.parent().change_debug)
            self.debug_label = QLabel('Use wandb: ')

            self.dataset_cbox = QComboBox()
            self.dataset_cbox.addItems(['<Not selected>', 'SWED sample', 'SWED FULL', 'SNOWED'])
            self.dataset_cbox.setCurrentText(self.parent().config['DATASET'])
            self.dataset_cbox.currentTextChanged.connect(self.parent().change_dataset)
            self.dataset_label = QLabel('Dataset: ')

            self.loss_cbox = QComboBox()
            self.loss_cbox.addItems(['<Not selected>', 'Crossentropy', 'Dice+Crossentropy', 'SeNetLoss'])
            self.loss_cbox.setCurrentText(self.parent().config['LOSS'])
            self.loss_cbox.currentTextChanged.connect(self.parent().change_loss)
            self.loss_label = QLabel('Loss function: ')

            self.optimizer_cbox = QComboBox()
            self.optimizer_cbox.addItems(['<Not selected>', 'Adam', 'SGD'])
            self.optimizer_cbox.setCurrentText(self.parent().config['OPTIMIZER'])
            self.optimizer_cbox.currentTextChanged.connect(self.parent().change_optimizer)
            self.optimizer_label = QLabel('Optimizer: ')

            self.momentum_sbox = QDoubleSpinBox()
            self.momentum_sbox.setRange(0, 1)
            self.momentum_sbox.setDecimals(2)
            self.momentum_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.momentum_sbox.setValue(self.parent().config['MOMENTUM'])
            self.momentum_sbox.valueChanged.connect(self.parent().change_momentum)
            self.momentum_label = QLabel('Optimizer momentum: ')

            self.weight_decay_sbox = QDoubleSpinBox()
            self.weight_decay_sbox.setRange(0, 1)
            self.weight_decay_sbox.setDecimals(2)
            self.weight_decay_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.weight_decay_sbox.setValue(self.parent().config['WEIGHT_DECAY'])
            self.weight_decay_sbox.valueChanged.connect(self.parent().change_weight_decay)
            self.weight_decay_label = QLabel('Optimizer weight decay: ')

            self.eval_freq_sbox = QSpinBox()
            self.eval_freq_sbox.setRange(0, self.parent().config['EPOCHS'])
            self.eval_freq_sbox.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
            self.eval_freq_sbox.setValue(self.parent().config['EVAL_FREQ'])
            self.eval_freq_sbox.valueChanged.connect(self.parent().change_eval_freq)
            self.eval_freq_label = QLabel('Number of epochs between metric evaluation: ')

            layout.addWidget(self.epochs_sbox, 6, 1)
            layout.addWidget(self.batchsize_sbox, 7, 1)
            layout.addWidget(self.lrate_sbox, 8, 1)
            layout.addWidget(self.train_part_sbox, 9, 1)
            layout.addWidget(self.scheduler_cbox, 10, 1)
            layout.addWidget(self.debug_cbox, 11, 1)
            layout.addWidget(self.dataset_cbox, 12, 1)
            layout.addWidget(self.loss_cbox, 13, 1)
            layout.addWidget(self.optimizer_cbox, 14, 1)
            layout.addWidget(self.momentum_sbox, 15, 1)
            layout.addWidget(self.weight_decay_sbox, 16, 1)
            layout.addWidget(self.eval_freq_sbox, 17, 1)

            layout.addWidget(self.epochs_label, 6, 0)
            layout.addWidget(self.batchsize_label, 7, 0)
            layout.addWidget(self.lrate_label, 8, 0)
            layout.addWidget(self.train_part_label, 9, 0)
            layout.addWidget(self.scheduler_label, 10, 0)
            layout.addWidget(self.debug_label, 11, 0)
            layout.addWidget(self.dataset_label, 12, 0)
            layout.addWidget(self.loss_label, 13, 0)
            layout.addWidget(self.optimizer_label, 14, 0)
            layout.addWidget(self.momentum_label, 15, 0)
            layout.addWidget(self.weight_decay_label, 16, 0)
            layout.addWidget(self.eval_freq_label, 17, 0)
            

        wid.setLayout(layout)

    def closeEvent(self, event):
        # return super().closeEvent(a0)
        print(self.parent().config)
        event.accept()