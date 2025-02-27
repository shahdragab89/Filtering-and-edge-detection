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
        
        # Initialize histograms
        original_hist = None
        equalized_hist = []
        
        # Check if image is grayscale or color
        if len(self.image.shape) == 2:
            # Grayscale image equalization
            hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            original_hist = hist.ravel() / hist.sum()  # Normalize
            
            # Cumulative Distribution Function (CDF)
            cdf = original_hist.cumsum()
            
            # Lookup table
            lut = np.round(cdf * max_value).astype(np.uint8)
            
            # Apply LUT
            equalized_image = cv2.LUT(self.image, lut)
            equalized_hist = [cv2.calcHist([equalized_image], [0], None, [256], [0, 256])]  # Wrap in a list for consistency
            
        else:
            # Color image equalization
            equalized_image = np.zeros_like(self.image)
            
            for channel in range(3):
                hist = cv2.calcHist([self.image], [channel], None, [256], [0, 256])
                original_hist = hist.ravel() / hist.sum()  # Normalize
                cdf = original_hist.cumsum()
                lut = np.round(cdf * max_value).astype(np.uint8)
                
                equalized_image[:, :, channel] = cv2.LUT(self.image[:, :, channel], lut)
                channel_hist = cv2.calcHist([equalized_image], [channel], None, [256], [0, 256])
                equalized_hist.append(channel_hist)
        
        # Print original and equalized histograms
        self.print_histograms(original_hist, equalized_hist)
        
        # Plot and display the equalized histogram
        self.plot_equalized_histogram(equalized_image, equalized_hist)
        
        # Display the equalized image in the provided label
        self.display_equalized_image(equalized_image)
        
        return equalized_image

    def print_histograms(self, original_hist, equalized_hist):
        # Print original histogram
        print("Original Histogram:")
        for value, frequency in enumerate(original_hist):
            print(f"Pixel Value: {value:3d}, Frequency: {frequency:.6f}")
        
        # Print equalized histogram
        print("\nEqualized Histogram:")
        if len(equalized_hist) == 1:  # Grayscale case
            for value in range(256):
                frequency = equalized_hist[0][value][0]
                print(f"Pixel Value: {value:3d}, Frequency: {frequency:.0f}")
        else:  # Color case
            for value in range(256):
                frequency = sum(channel[value][0] for channel in equalized_hist)
                print(f"Pixel Value: {value:3d}, Frequency: {frequency:.0f}")

        # Summary statistics for original histogram
        print("\nOriginal Histogram Summary:")
        print(f"Min Frequency: {original_hist.min():.6f}, Max Frequency: {original_hist.max():.6f}, "
              f"Mean Frequency: {original_hist.mean():.6f}, Std Dev: {original_hist.std():.6f}")

        # Summary statistics for equalized histogram
        equalized_freq = np.concatenate(equalized_hist).flatten() if len(equalized_hist) > 1 else equalized_hist[0].flatten()
        print("\nEqualized Histogram Summary:")
        print(f"Min Frequency: {equalized_freq.min():.0f}, Max Frequency: {equalized_freq.max():.0f}, "
              f"Mean Frequency: {equalized_freq.mean():.0f}, Std Dev: {equalized_freq.std():.0f}")

    def plot_equalized_histogram(self, equalized_image, equalized_hist):
        plt.figure(figsize=(4, 3), dpi=100)

        if len(equalized_hist) == 1:  # Grayscale case
            plt.plot(equalized_hist[0], color='black')
            plt.title("Equalized Grayscale Histogram (CDF)")
        else:  # Color case
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.plot(equalized_hist[i], color=color)
            plt.title("Equalized RGB Histogram (CDF)")
        
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, 5000])  # Set Y-axis limits to 0 to 25,000
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
            # Color image - convert from BGR (OpenCV) to RGB (Qt)
            rgb_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.equalized_image_label.setPixmap(pixmap)
        self.equalized_image_label.setScaledContents(True)