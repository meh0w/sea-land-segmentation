from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QMainWindow, QFileDialog, QComboBox, QGridLayout
from matplotlib.figure import Figure

from gui.inference_window import Inference_Window
from gui.train_window import Train_Window

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Sea-Land Segmentation GUI"

        self.setFixedSize(200, 200)

        self.train_button = QPushButton('Training', self)
        self.train_button.move(50, 50)

        self.inference_button = QPushButton('Inference', self)
        self.inference_button.move(50, 100)

        self.train_window = Train_Window(self)
        self.train_button.clicked.connect(self.show_train_window)

        self.inference_window = Inference_Window(self)
        self.inference_button.clicked.connect(self.show_inference_window)

        self.show_main_window()

    def show_train_window(self):
        if self.train_window.isHidden():
            self.train_window.show()
            self.hide()

    def show_inference_window(self):
        if self.inference_window.isHidden():
            self.inference_window.show()
            self.hide()

    def show_main_window(self):
        self.setWindowTitle(self.title)
        self.show()

