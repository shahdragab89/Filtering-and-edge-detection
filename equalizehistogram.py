import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

class Equalize_Histogram:
    def __init__(self, image, label: QLabel, equalized_image_label: QLabel):
        if image is None or isinstance(image, bool): 
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            return 
        self.image = image 
        self.label = label  
        self.equalized_image_label = equalized_image_label  
        self.equalized_image = self.equalize_histogram()

    def equalize_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram equalization.")
            return None
        
        #(assuming 8-bit by default) (0-255)
        bit_depth = 8
        max_value = (2 ** bit_depth) - 1  
        
        original_hist = None
        equalized_hist = []
        

        hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        original_hist = hist.ravel() / hist.sum()  #normalize
            
        #cdf
        cdf = original_hist.cumsum()
            
        #look up table
        lut = np.round(cdf * max_value).astype(np.uint8)

        equalized_image = cv2.LUT(self.image, lut)
        
        eq_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
        equalized_hist = eq_hist.ravel() / eq_hist.sum()            

        #print statments for CHECKING!!!!
        #self.print_histograms(original_hist, equalized_hist)
        
        self.plot_equalized_histogram(equalized_image, equalized_hist)
        self.display_equalized_image(equalized_image)
        
        return equalized_image

    # def print_histograms(self, original_hist, equalized_hist):
    #     #print statments for CHECKING!!!!
    #     print("Original Histogram:")
    #     for value, frequency in enumerate(original_hist):
    #         print(f"Pixel Value: {value:3d}, Frequency: {frequency:.6f}")
        
    #     #print statments for CHECKING!!!!
    #     print("\nEqualized Histogram:")
    #     for value in range(256):
    #             frequency = equalized_hist[0][value][0]
    #             print(f"Pixel Value: {value:3d}, Frequency: {frequency:.0f}")

    #     #print statments for CHECKING!!!!
    #     print("\nOriginal Histogram Summary:")
    #     print(f"Min Frequency: {original_hist.min():.6f}, Max Frequency: {original_hist.max():.6f}, "
    #           f"Mean Frequency: {original_hist.mean():.6f}, Std Dev: {original_hist.std():.6f}")

    #     #print statments for CHECKING!!!!
    #     equalized_freq = np.concatenate(equalized_hist).flatten() if len(equalized_hist) > 1 else equalized_hist[0].flatten()
    #     print("\nEqualized Histogram Summary:")
    #     print(f"Min Frequency: {equalized_freq.min():.0f}, Max Frequency: {equalized_freq.max():.0f}, "
    #           f"Mean Frequency: {equalized_freq.mean():.0f}, Std Dev: {equalized_freq.std():.0f}")

    def plot_equalized_histogram(self, equalized_image, equalized_hist):
        plt.figure(figsize=(4, 3), dpi=100)
        plt.plot(equalized_hist[0], color='black')
        plt.title("Equalized Grayscale Histogram (CDF)")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, 5000])  
        plt.grid()
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        image = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(image)

        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
    
    def display_equalized_image(self, equalized_image):
        if self.equalized_image_label is None:
            print("No QLabel provided to display equalized image.")
            return
        
        #numpy --> qimage
        height, width = equalized_image.shape
        bytes_per_line = width
        q_image = QImage(equalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        #qimage --> pixmap
        pixmap = QPixmap.fromImage(q_image)
        self.equalized_image_label.setPixmap(pixmap)
        self.equalized_image_label.setScaledContents(True)