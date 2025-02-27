import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from io import BytesIO

class Normalize_Histogram:
    def __init__(self, image, label: QLabel, normalized_image_label: QLabel):
        # Check if the image is a valid grayscale image
        if image is None or isinstance(image, bool) or len(image.shape) != 2:
            print("Error: Invalid image provided for histogram computation.")
            self.image = None
            self.label = None
            self.normalized_image_label = None
            return

        print(f"Input image shape: {image.shape}")  # Debug info
        print(f"Unique pixel values: {np.unique(image)}")  # Debug info

        self.image = image  # Store the image
        self.label = label  # QLabel to display the histogram
        self.normalized_image_label = normalized_image_label  # QLabel to display the normalized image
        self.normalized_image = self.normalize_histogram()

    def normalize_histogram(self):
        if self.image is None:
            print("No valid image loaded for histogram normalization.")
            return None
        
        # Normalize the grayscale image
        normalized_image = self.normalize_grayscale(self.image)

        # Plot and display the normalized histogram if label is provided
        if self.label is not None:
            self.plot_normalized_histogram(normalized_image)
        
        # Display the normalized image in the provided label if available
        if self.normalized_image_label is not None:
            self.display_normalized_image(normalized_image)
        
        return normalized_image
    
    def normalize_grayscale(self, img):
        min_val = np.min(img)
        max_val = np.max(img)

        print(f"Min value: {min_val}, Max value: {max_val}")  # Debug info
        
        if min_val < max_val:
            normalized_img = ((img - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            print("All pixel values are the same. Returning original image.")
            normalized_img = img.copy()
        
        return normalized_img

    def plot_normalized_histogram(self, normalized_image):
        if self.label is None:
            return
        
        plt.figure(figsize=(4, 3), dpi=100)
        plt.hist(normalized_image.ravel(), bins=256, color='black', alpha=0.7)
        plt.title("Normalized Grayscale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.grid()
        
        # Save the plot to a buffer
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        plt.close()  # Explicitly close the figure to prevent memory leaks
        
        # Convert buffer image to QPixmap
        buffer.seek(0)
        image = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(image)
        
        # Display in QLabel
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)
        
    def display_normalized_image(self, normalized_image):
        if self.normalized_image_label is None:
            print("No QLabel provided to display normalized image.")
            return
        
        # Convert the numpy array to QImage
        height, width = normalized_image.shape
        bytes_per_line = width
        q_image = QImage(normalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.normalized_image_label.setPixmap(pixmap)
        self.normalized_image_label.setScaledContents(True)
