import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO
from histogram import calc_hist

def apply_LUT(image, lookup_table):
    result = np.zeros_like(image)
    height, width = image.shape[:2]
    #apply lut to each pixel
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            result[y, x] = lookup_table[pixel_value]
    return result

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
        
        hist = calc_hist([self.image], [0], None, [256], [0, 256])
        original_hist = hist.ravel() / hist.sum()  #normalize
            
        #cdf
        cdf = original_hist.cumsum()
            
        #look up table
        lut = np.round(cdf * max_value).astype(np.uint8)

        equalized_image = apply_LUT(self.image, lut)
        equalized_hist = [calc_hist([equalized_image], [0], None, [256], [0, 256])]  
        
        
        self.plot_equalized_histogram(equalized_image, equalized_hist)
        self.display_equalized_image(equalized_image)
        
        return equalized_image

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