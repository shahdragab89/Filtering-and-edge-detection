import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

class Histogram:
    def __init__(self, image, label: QLabel):
        if image is None or isinstance(image, bool):  # Handle invalid images
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            return
        
        self.image = image  # Store the image
        self.label = label  # QLabel to display the histogram
        self.plot_histogram()

    def plot_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram computation.")
            return
        
        # Check if image is grayscale or color
        if len(self.image.shape) == 2:
            # Grayscale image - plot single histogram
            hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            
            plt.figure(figsize=(4, 3), dpi=100)
            plt.plot(hist, color='black')
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.grid()
            
        else:
            # Color image - plot histogram for each channel
            colors = ('b', 'g', 'r')
            plt.figure(figsize=(4, 3), dpi=100)
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            
            plt.title("RGB Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.grid()
        
        # Save the plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        plt.close()
        
        # Convert buffer image to QPixmap
        buffer.seek(0)
        image = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(image)
        
        # Display in QLabel
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
