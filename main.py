from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QPixmap, QImage
import sys
from PIL import Image
import numpy as p
import cv2
import random
import os
import matplotlib.pyplot as plt
from noise import NoiseProcessor
from filters import FilterProcessor


# Load the UI file
ui, _ = loadUiType("allUi.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        # Initializing Buttons 
        self.filterUpload_button.clicked.connect(lambda: self.uploadImage(1))
        self.filterDownload_button.clicked.connect(self.downloadImage)
        self.apply_button.clicked.connect(self.apply)
        self.reset_button.clicked.connect(lambda: self.reset(1))

        # Initializing ComboBoxes
        self.noise_comboBox.currentIndexChanged.connect(self.handleNoise)
        self.filter_comboBox.currentIndexChanged.connect(self.handleFilter)

        # Initializing Sliders
        self.kernel_slider.sliderReleased.connect(lambda: self.handleFilter("kernel"))
        self.sigma_slider.sliderReleased.connect(lambda: self.handleFilter("sigma"))
        self.mean_slider.sliderReleased.connect(lambda: self.handleFilter("mean"))

        # Allow scaling of image
        self.original_image.setScaledContents(True)  
        self.filtered_image.setScaledContents(True)


    def uploadImage(self, value):
        # Value defines which label to show the picture on 
        self.value = value
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Get the dimension of the image
            height, width = self.image.shape
            bytes_per_line = width

            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            self.original_image.setPixmap(QPixmap.fromImage(q_image))
            self.original_image.setScaledContents(True)
            match value:
                case 1:
                    # Convert QImage to QPixmap and set it to the QLabel
                    self.original_image.setPixmap(QPixmap.fromImage(q_image))
                    self.original_image.setScaledContents(True)  # Scale the image to fit QLabel
                    self.filtered_image.setPixmap(QPixmap.fromImage(q_image))
                    self.filtered_image.setScaledContents(True)  
                case 2:
                    self.rgbOriginal_image.setPixmap(QPixmap.fromImage(q_image))
                    self.rgbOriginal_image.setScaledContents(True)  
                case 3:
                    self.histogramOriginal_image.setPixmap(QPixmap.fromImage(q_image))
                    self.histogramOriginal_image.setScaledContents(True)  
                case 4:
                    self.image1.setPixmap(QPixmap.fromImage(q_image))
                    self.image1.setScaledContents(True)  
                case 5:
                    self.image2.setPixmap(QPixmap.fromImage(q_image))
                    self.image2.setScaledContents(True)                
        print("upload")

    def downloadImage(self):
        print("download")

    def handleNoise(self):
        noisyImage = NoiseProcessor.applyNoiseAndDisplay(self.noise_comboBox.currentText(), self.image)
        self.noisyImage = noisyImage
        self.filtered_image.setPixmap(QPixmap.fromImage(noisyImage))
        self.filtered_image.setScaledContents(True)  
    
    def handleFilter(self, updatedSlider):
        self.updatedSlider = updatedSlider
        match self.updatedSlider:
            case "kernel":
                self.kernel_value = self.kernel_slider.value()
                self.kernel_label.setText(str(self.kernel_value))
            case "sigma":
                self.sigma_value = self.sigma_slider.value()
                self.sigma_label.setText(str(self.sigma_value))
            case "mean":
                self.mean_value = self.mean_slider.value()
                self.mean_label.setText(str(self.mean_value))
        self.sliderValues = [self.kernel_slider.value(), self.sigma_slider.value(), self.mean_slider.value()]
        
        filteredImage = FilterProcessor.applyFilterAndDisplay(self.filter_comboBox.currentText(), self.sliderValues, self.image)
        self.filteredImage = filteredImage
        self.filtered_image.setPixmap(QPixmap.fromImage(filteredImage))
        self.filtered_image.setScaledContents(True) 


    def apply(self):
        self.original_image.setPixmap(QPixmap.fromImage(self.noisyImage))
        self.original_image.setScaledContents(True)  
        

    def reset(self, value):
        # Value defines which page to remove the picture from
        self.value = value
        if self.value == 1:
            self.image = None
            self.original_image.clear() 
            self.filtered_image.clear() 
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
