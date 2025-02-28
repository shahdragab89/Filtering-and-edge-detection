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

def apply_LUT(image, lookup_table):
    """
    Custom implementation of Look-Up Table (LUT) without using cv2.LUT
    
    Parameters:
    image: Input image
    lookup_table: Array of 256 elements representing the mapping
    
    Returns:
    result: Image after applying the LUT
    """
    # Create a copy of the input image
    result = np.zeros_like(image)
    
    # Apply the lookup table to each pixel
    height, width = image.shape[:2]
    
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            result[y, x] = lookup_table[pixel_value]
    
    return result

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
        
        # Use custom calc_hist instead of cv2.calcHist
        hist = calc_hist([self.image], [0], None, [256], [0, 256])
        original_hist = hist.ravel() / hist.sum()  #normalize
            
        #cdf
        cdf = original_hist.cumsum()
            
        #look up table
        lut = np.round(cdf * max_value).astype(np.uint8)

        # Use custom apply_LUT instead of cv2.LUT
        equalized_image = apply_LUT(self.image, lut)
        
        # Use custom calc_hist for the equalized image too
        equalized_hist = [calc_hist([equalized_image], [0], None, [256], [0, 256])]  # Wrap in a list for consistency
            
        #print statments for CHECKING!!!!
        self.print_histograms(original_hist, equalized_hist)
        
        self.plot_equalized_histogram(equalized_image, equalized_hist)
        self.display_equalized_image(equalized_image)
        
        return equalized_image

    def print_histograms(self, original_hist, equalized_hist):
        #print statments for CHECKING!!!!
        print("Original Histogram:")
        for value, frequency in enumerate(original_hist):
            print(f"Pixel Value: {value:3d}, Frequency: {frequency:.6f}")
        
        #print statments for CHECKING!!!!
        print("\nEqualized Histogram:")
        for value in range(256):
                frequency = equalized_hist[0][value][0]
                print(f"Pixel Value: {value:3d}, Frequency: {frequency:.0f}")

        #print statments for CHECKING!!!!
        print("\nOriginal Histogram Summary:")
        print(f"Min Frequency: {original_hist.min():.6f}, Max Frequency: {original_hist.max():.6f}, "
              f"Mean Frequency: {original_hist.mean():.6f}, Std Dev: {original_hist.std():.6f}")

        #print statments for CHECKING!!!!
        equalized_freq = np.concatenate(equalized_hist).flatten() if len(equalized_hist) > 1 else equalized_hist[0].flatten()
        print("\nEqualized Histogram Summary:")
        print(f"Min Frequency: {equalized_freq.min():.0f}, Max Frequency: {equalized_freq.max():.0f}, "
              f"Mean Frequency: {equalized_freq.mean():.0f}, Std Dev: {equalized_freq.std():.0f}")

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