from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QPixmap, QImage
import sys
from PIL import Image
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from noise import NoiseProcessor
from filters import FilterProcessor
from PIL import Image, ImageQt
import traceback
from edgedetectors import EdgeDetector
from PyQt5.QtCore import QBuffer, QIODevice
import PIL.ImageQt as ImageQtModule
from histogram import Histogram
from equalizehistogram import Equalize_Histogram
from normalizehistogram import Normalize_Histogram


# Manually patch QBuffer and QIODevice into ImageQt
ImageQtModule.QBuffer = QBuffer
ImageQtModule.QIODevice = QIODevice




# Load the UI file
ui, _ = loadUiType("allUi.ui")

class MainApp(QtWidgets.QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.image = None
        self.value = None
        self.sigma_value = 0
        # Initializing Buttons 
        self.filterUpload_button.clicked.connect(lambda: self.uploadImage(1))
        self.filterDownload_button.clicked.connect(self.downloadImage)
        self.rgbUpload_button.clicked.connect(lambda: self.uploadImage(2))
        self.upload_button_2.clicked.connect(lambda: self.uploadImage(3))
        self.upload_image1_button.clicked.connect(lambda: self.uploadImage(4))
        self.upload_image2_button.clicked.connect(lambda: self.uploadImage(5))
        self.rgbDownload_button.clicked.connect(self.downloadImage)
        #self.download_equalizer.clicked.connect(Equalize_Histogram.equalize_histogram)
        self.download_normalized.clicked.connect(self.downloadImage)

        self.apply_button.clicked.connect(self.apply)
        self.reset_button.clicked.connect(lambda: self.reset(1))

        # Initializing ComboBoxes
        self.noise_comboBox.activated.connect(self.handleNoise)
        self.filter_comboBox.activated.connect(self.handleFilter)

        # Initializing Sliders
        self.kernel_slider.sliderReleased.connect(self.handleFilter)
        self.sigma_slider.sliderReleased.connect(self.handleFilter)
        self.mean_slider.sliderReleased.connect(self.handleNoise)

        # Allow scaling of image
        self.original_image.setScaledContents(True)  
        self.filtered_image.setScaledContents(True)

        # Initialize the edge detection combobox
        self.edges_comboBox.activated.connect(self.handleEdgeDetection)



    def uploadImage(self, value):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Get the dimension of the image
            height, width = self.image.shape
            bytes_per_line = width

            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            self.value = value

            match value:
                case 1:
                    self.original_image.setPixmap(QPixmap.fromImage(q_image))
                    self.filtered_image.setPixmap(QPixmap.fromImage(q_image))  # If needed for filtering
                case 2:

                    self.rgbOriginal_image.setPixmap(QPixmap.fromImage(q_image))
                case 3:
                    self.histogramOriginal_image.setPixmap(QPixmap.fromImage(q_image))
                    self.histogram = Histogram(self.image, self.label_51)
                    self.equalizehistogram = Equalize_Histogram(self.image, self.label_53, self.equalized_image)
                    self.normalizehistogram = Normalize_Histogram(self.image, self.label_55, self.normalized_image)              
           
                case 4:
                    self.image1.setPixmap(QPixmap.fromImage(q_image))
                case 5:
                    self.image2.setPixmap(QPixmap.fromImage(q_image))

            # Set scaled contents for each QLabel only once
            self.original_image.setScaledContents(True)
            self.filtered_image.setScaledContents(True)
            self.rgbOriginal_image.setScaledContents(True)
            self.histogramOriginal_image.setScaledContents(True)
            self.image1.setScaledContents(True)
            self.image2.setScaledContents(True)

        print("upload")

    def downloadImage(self):
        if self.value is None:
            print("No image uploaded yet. Please upload an image before downloading.")
            return
        
        # Mapping value to QLabel attributes
        image_mapping = {
            1: self.original_image,
            2: self.rgbOriginal_image,
            3: self.histogramOriginal_image,
            4: self.image1,
            5: self.image2
        }

        label = image_mapping.get(self.value)

        if not label or label.pixmap() is None:
            print("No valid image found to download.")
            return
        
        pixmap = label.pixmap()

        # Open save dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.bmp *.jpeg)", options=options)
        
        if file_path:
            if pixmap and not pixmap.isNull():
                pixmap.save(file_path)
                print(f"Image saved to {file_path}")
            else:
                print("No valid image found in QLabel.")

    def handleNoise(self):
        try:
            if self.image is None:
                self.mean_slider.setValue(1)
                raise ValueError("No image loaded. Please upload an image before applying noise.")
            # Take the mean value for gaussian noise
            self.mean_value = self.handle_kernelSlider()[2]

            noisyImage = NoiseProcessor.applyNoiseAndDisplay(self.noise_comboBox.currentText(), self.image, self.mean_value)
            self.noisyImage = noisyImage

            # Show the noisy image 
            self.filtered_image.setPixmap(QPixmap.fromImage(noisyImage))
            self.filtered_image.setScaledContents(True)  

        except ValueError as ve:
            print(f" Error: {ve}")

    
    def handleFilter(self):
        try:
            if self.image is None:
                self.kernel_slider.setValue(3)
                self.sigma_slider.setValue(1)
                raise ValueError("No image loaded. Please upload an image before applying noise.")

            self.sliderValues = self.handle_kernelSlider()

            # Apply filter to noisy image
            if hasattr(NoiseProcessor, 'last_noisy_image') and NoiseProcessor.last_noisy_image is not None:
                noisyImage = NoiseProcessor.last_noisy_image  
            else:
                noisyImage = self.image 
            
            filteredImage = FilterProcessor.applyFilterAndDisplay(noisyImage, self.filter_comboBox.currentText(), self.sliderValues)
            self.filteredImage = filteredImage
            self.filtered_image.setPixmap(QPixmap.fromImage(filteredImage))
            self.filtered_image.setScaledContents(True) 

        except ValueError as ve:
            print(f" Error: {ve}")

    def handle_kernelSlider(self):
    
        kernel_allowedValues = [1,3,5,7,9,11,13]
        self.kernel_value = kernel_allowedValues[self.kernel_slider.value()]
        self.kernel_label.setText(str(self.kernel_value))
        self.sigma_value = self.sigma_slider.value()
        self.sigma_label.setText(str(self.sigma_value))
        self.mean_value = self.mean_slider.value()
        self.mean_label.setText(str(self.mean_value))

        return [self.kernel_value, self.sigma_value, self.mean_value]

    def apply(self):
        try:
            if self.image is None:
                raise ValueError("No image loaded. Please upload an image before applying noise.")
            # Set the noisy image in the original_image label
            self.original_image.setPixmap(QPixmap.fromImage(self.noisyImage))
            self.original_image.setScaledContents(True)  
        except ValueError as ve:
            print(f" Error: {ve}")
        

    def reset(self, value):
        # Value defines which page to remove the picture from
        self.value = value
        if self.value == 1:
            self.image = None
            self.original_image.clear() 
            self.filtered_image.clear() 
        
    def handleEdgeDetection(self):
        try:
            if self.image is None:
                raise ValueError("No image loaded. Please upload an image before applying edge detection.")

            # Get the selected edge detection method
            method = self.edges_comboBox.currentText()
            
            # If "None" is selected, just display the current image
            if method == "None":
                # Display the current image without edge detection
                if hasattr(self, 'filteredImage'):
                    self.filtered_image.setPixmap(QPixmap.fromImage(self.filteredImage))
                elif hasattr(self, 'noisyImage'):
                    self.filtered_image.setPixmap(QPixmap.fromImage(self.noisyImage))
                else:
                    # Display original image
                    height, width = self.image.shape
                    bytes_per_line = width
                    q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    self.filtered_image.setPixmap(QPixmap.fromImage(q_image))
                return

            # Get threshold values for Canny (used only for Canny method)
            # threshold1 = self.threshold1_slider.value()
            # threshold2 = self.threshold2_slider.value()
            threshold1 = 50
            threshold2 = 150
            sigma = self.sigma_value
            
            # Update threshold labels
            # self.threshold1_label.setText(str(threshold1))
            # self.threshold2_label.setText(str(threshold2))
            
            # Get kernel size for Sobel and Canny
            kernel_values = [1, 3, 5, 7]
            aperture_size = kernel_values[self.kernel_slider.value() % len(kernel_values)]
            
            # Get the base image to apply edge detection on
            if hasattr(self, 'filteredImage'):
                # Use the filtered image if available
                input_image = ImageQt.fromqimage(self.filteredImage).convert('L')
                input_array = np.array(input_image)
            elif hasattr(self, 'noisyImage'):
                # Use the noisy image if available
                input_image = ImageQt.fromqimage(self.noisyImage).convert('L')
                input_array = np.array(input_image)
            else:
                # Use the original image
                input_array = self.image
            
            # Apply edge detection
            edge_image = EdgeDetector.apply_edge_detection(
                input_array, 
                method, 
                threshold1, 
                threshold2, 
                aperture_size,
                sigma
            )
            
            if edge_image is None:
                print("Edge detection returned None. Displaying an error message or a blank image.")
                # Optionally display a blank image or an error message
            else:
                self.edgeImage = edge_image
                self.filtered_image.setPixmap(QPixmap.fromImage(edge_image))
                self.filtered_image.setScaledContents(True)

            
        except ValueError as ve:
            print(f"Error: {ve}")
            # You might want to add a more user-friendly error notification here
            

        except Exception as e:
            print("Unexpected error in edge detection:")
            traceback.print_exc()  # This prints the full error traceback


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
