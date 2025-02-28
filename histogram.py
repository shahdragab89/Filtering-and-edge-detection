import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

def calc_hist(images, channels, mask, histSize, ranges):
    """
    Custom implementation of histogram calculation without using cv2.calcHist
    
    Parameters:
    images: List containing input image(s)
    channels: List of channels for which to compute histogram 
    mask: Optional mask
    histSize: List containing number of bins
    ranges: List containing min and max value for histogram
    
    Returns:
    hist: Computed histogram as numpy array
    """
    # Extract the image from the list (like cv2.calcHist expects)
    image = images[0]
    
    # Extract parameters
    bins = histSize[0]
    min_val, max_val = ranges
    
    # Initialize histogram with zeros
    hist = np.zeros(bins, dtype=np.float32)
    
    # Calculate bin width
    bin_width = (max_val - min_val) / bins
    
    # Flatten the image if it's not already 1D
    if len(image.shape) > 2:
        # If it's a color image, use the specified channel
        channel = channels[0]
        if image.shape[2] <= channel:
            raise ValueError(f"Channel {channel} does not exist in the image")
        flat_image = image[:,:,channel].flatten()
    else:
        # It's a grayscale image
        flat_image = image.flatten()
    
    # Apply mask if provided
    if mask is not None:
        flat_mask = mask.flatten()
        flat_image = flat_image[flat_mask != 0]
    
    # Count pixels in each bin
    for pixel_value in flat_image:
        # Calculate bin index
        bin_idx = int((pixel_value - min_val) / bin_width)
        
        # Ensure the index is within range
        if 0 <= bin_idx < bins:
            hist[bin_idx] += 1
    
    # Reshape to match cv2.calcHist output format
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
        
        #show CDF in a separate window
        self.plot_cdf_popup()

    def plot_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram computation.")
            return
        
        # Use our custom calc_hist function instead of cv2.calcHist
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
    
    def plot_cdf_popup(self):
        if self.image is None:
            print("No valid image loaded for CDF computation.")
            return
        plt.figure(figsize=(8, 6))
        
        # Use our custom calc_hist function instead of cv2.calcHist
        hist = calc_hist([self.image], [0], None, [256], [0, 256])
        
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