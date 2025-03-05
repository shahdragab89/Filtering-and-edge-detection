import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5 import Qt
from io import BytesIO

def calc_hist(images, channels, mask, histSize, ranges):
    image = images[0] #select first image of the list
    bins = histSize[0]
    min_val, max_val = ranges
    
    #initialize histogram with zeros
    hist = np.zeros(bins, dtype=np.float32)
    
    #calculate bin width
    bin_width = (max_val - min_val) / bins
    
    #flatten the image to 1D
    if len(image.shape) > 2:
        # If it's a color image, use the specified channel
        channel = channels[0]
        if image.shape[2] <= channel:
            raise ValueError(f"Channel {channel} does not exist in the image")
        flat_image = image[:,:,channel].flatten()
    else:
        # It's a grayscale image
        flat_image = image.flatten()
    
    for pixel_value in flat_image:
        #calculate bin index
        bin_idx = int((pixel_value - min_val) / bin_width)
        #ensure idx is within range
        if 0 <= bin_idx < bins:
            hist[bin_idx] += 1
    hist = hist.reshape(-1, 1)
    return hist

class Histogram:
    def __init__(self, image, label: QLabel):
        if image is None or isinstance(image, bool): 
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            return
        
        self.image = image 
        self.label = label  
        self.plot_histogram()

    def plot_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram computation.")
            return
        
        hist = calc_hist([self.image], [0], None, [256], [0, 256])
            
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
        
        #convert buffer image to qpixmap & show in qlabel
        buffer.seek(0)
        image = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
