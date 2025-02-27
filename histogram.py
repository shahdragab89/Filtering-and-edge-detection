import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

class Histogram:
    def __init__(self, image, label: QLabel):
        if image is None or isinstance(image, bool): 
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            return
        
        self.image = image 
        self.label = label  
        self.plot_histogram()
        
        #show CDF in a separate window
        self.plot_cdf_popup()

    def plot_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram computation.")
            return
        

        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            
        plt.figure(figsize=(4, 3), dpi=100)
        plt.plot(hist, color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, hist.max() * 1.1])
        plt.grid()
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        plt.close()
        
        #convert buffer image to qpixmap & shsow in qlabel
        buffer.seek(0)
        image = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
    
    def plot_cdf_popup(self):
        if self.image is None:
            print("No valid image loaded for CDF computation.")
            return
        plt.figure(figsize=(8, 6))
        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)
        plt.plot(cdf, color='black')
        plt.title("Grayscale CDF Distribution")
        plt.xlabel("Pixel Value")
        plt.ylabel("Cumulative Probability")
        plt.xlim([0, 256])
        plt.ylim([0, 1.05])
        plt.grid()
        plt.show(block=False)