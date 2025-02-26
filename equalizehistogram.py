import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

class Equalize_Histogram:
    def __init__(self, image, label: QLabel, equalized_image_label: QLabel):
        if image is None or isinstance(image, bool):  # Handle invalid images
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            return
        
        self.image = image  # Store the image
        self.label = label  # QLabel to display the histogram
        self.equalized_image_label = equalized_image_label  # QLabel to display the equalized image
        self.equalized_image = self.equalize_histogram()

    def equalize_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram equalization.")
            return None
        
        # Determine bit depth (assuming 8-bit by default)
        bit_depth = 8
        max_value = (2 ** bit_depth) - 1  # Usually 255 for 8-bit images
        
        # Check if image is grayscale or color
        if len(self.image.shape) == 2:
            # Grayscale image equalization using CDF
            # Calculate histogram
            hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()  # Normalize
            
            # Calculate cumulative distribution function (CDF)
            cdf = hist.cumsum()
            
            # Create lookup table
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                lut[i] = np.round(cdf[i] * max_value)
            
            # Apply lookup table (map pixels)
            equalized_image = cv2.LUT(self.image, lut)
            
            # Calculate histogram of equalized image
            equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
            
        else:
            # Color image equalization - apply to each channel separately
            equalized_image = np.zeros_like(self.image)
            equalized_hist = []
            
            # Process each channel independently
            for channel in range(3):
                # Calculate histogram for this channel
                hist = cv2.calcHist([self.image], [channel], None, [256], [0, 256])
                hist = hist.ravel() / hist.sum()  # Normalize
                
                # Calculate cumulative distribution function (CDF)
                cdf = hist.cumsum()
                
                # Create lookup table
                lut = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    lut[i] = np.round(cdf[i] * max_value)
                
                # Apply lookup table to this channel
                equalized_image[:,:,channel] = cv2.LUT(self.image[:,:,channel], lut)
                
                # Calculate histogram of equalized channel
                channel_hist = cv2.calcHist([equalized_image], [channel], None, [256], [0, 256])
                equalized_hist.append(channel_hist)
        
        # Plot and display the equalized histogram
        self.plot_equalized_histogram(equalized_image, equalized_hist)
        
        # Display the equalized image in the provided label
        self.display_equalized_image(equalized_image)
        
        return equalized_image

    def plot_equalized_histogram(self, equalized_image, equalized_hist):
        plt.figure(figsize=(4, 3), dpi=100)
        
        # Check if we're dealing with grayscale or color histogram
        if isinstance(equalized_hist, list):
            # Color image - plot histogram for each channel
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.plot(equalized_hist[i], color=color)
            plt.title("Equalized RGB Histogram (CDF)")
        else:
            # Grayscale image - plot single histogram
            plt.plot(equalized_hist, color='black')
            plt.title("Equalized Grayscale Histogram (CDF)")
        
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
    
    def display_equalized_image(self, equalized_image):
        if self.equalized_image_label is None:
            print("No QLabel provided to display equalized image.")
            return
        
        # Convert the numpy array to QImage
        if len(equalized_image.shape) == 2:
            # Grayscale image
            height, width = equalized_image.shape
            bytes_per_line = width
            q_image = QImage(equalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color image - need to convert from BGR (OpenCV) to RGB (Qt)
            rgb_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.equalized_image_label.setPixmap(pixmap)
        self.equalized_image_label.setScaledContents(True)