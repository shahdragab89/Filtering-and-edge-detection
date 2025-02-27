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
        
        # Automatically show CDF in a separate window
        self.plot_cdf_popup()

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
            plt.ylim([0, hist.max() * 1.1])
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
    
    def plot_cdf_popup(self):
        """Plot the CDF (Cumulative Distribution Function) in a separate matplotlib window"""
        if self.image is None:
            print("No valid image loaded for CDF computation.")
            return
        
        # Create a new figure for the CDF
        plt.figure(figsize=(8, 6))
        
        # Check if image is grayscale or color
        if len(self.image.shape) == 2:
            # Grayscale image - plot single CDF
            hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            # Normalize histogram to get PDF
            pdf = hist / hist.sum()
            # Calculate CDF from PDF
            cdf = np.cumsum(pdf)
            
            plt.plot(cdf, color='black')
            plt.title("Grayscale CDF Distribution")
            
        else:
            # Color image - plot CDF for each channel
            colors = ('b', 'g', 'r')
            labels = ('Blue', 'Green', 'Red')
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                # Normalize histogram to get PDF
                pdf = hist / hist.sum()
                # Calculate CDF from PDF
                cdf = np.cumsum(pdf)
                
                plt.plot(cdf, color=color, label=label)
            
            plt.legend()
            plt.title("RGB CDF Distribution")
        
        plt.xlabel("Pixel Value")
        plt.ylabel("Cumulative Probability")
        plt.xlim([0, 256])
        plt.ylim([0, 1.05])
        plt.grid()
        
        # Show the plot in a non-blocking way
        plt.show(block=False)