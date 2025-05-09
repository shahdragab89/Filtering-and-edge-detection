import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image


class GrayscaleProcessor:
    def __init__(self, ui):
        
        self.ui = ui
        self.original_image = None
        self.gray_image = None

    def load_image(self, file_path):
       
        self.original_image = Image.open(file_path).convert("RGB")
        self.display_original_image()

    def display_original_image(self):
       
        if self.original_image is None:
            return

        q_image = self.pil_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(q_image)

        self.ui.rgbOriginal_image.setPixmap(pixmap)
        self.ui.rgbOriginal_image.setScaledContents(True)

    def convert_to_grayscale(self):
       
        if self.original_image is None:
            return

        # Convert image to numpy array
        img_array = np.array(self.original_image)

        # Convert to grayscale using standard formula
        grayscale_values = (
            0.299 * img_array[:, :, 0] + 
            0.587 * img_array[:, :, 1] + 
            0.114 * img_array[:, :, 2]
        )
        self.gray_image = grayscale_values.astype(np.uint8)

        # Convert grayscale numpy array to QImage
        height, width = self.gray_image.shape
        bytes_per_line = width
        q_image = QImage(self.gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        self.ui.rgbGray_image.setPixmap(pixmap)
        self.ui.rgbGray_image.setScaledContents(True)

    def compute_histograms(self):
       
        if self.original_image is None:
            return

        img_array = np.array(self.original_image)
        colors = ['red', 'green', 'blue']

        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#454674")  
        ax.set_facecolor("#454674")  

        for i in range(3):  
            hist, bins = np.histogram(img_array[:, :, i], bins=256, range=(0, 256), density=True)
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]  
            ax.plot(bins[:-1], cdf, color=colors[i])

        ax.tick_params(axis='both', colors='white')

        plt.savefig("cdf_plot.png", bbox_inches='tight', facecolor="#454674")
        plt.close()
        self.ui.rgbCDF_image.setPixmap(QPixmap("cdf_plot.png"))
        self.ui.rgbCDF_image.setScaledContents(True)

        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#454674")
        ax.set_facecolor("#454674")  

        for i in range(3):  
            hist, bins = np.histogram(img_array[:, :, i], bins=256, range=(0, 256), density=True)
            ax.plot(bins[:-1], hist, color=colors[i])

        
        ax.tick_params(axis='both', colors='white')

        plt.savefig("pdf_plot.png", bbox_inches='tight', facecolor="#454674")
        plt.close()
        self.ui.rgbPDF_image.setPixmap(QPixmap("pdf_plot.png"))
        self.ui.rgbPDF_image.setScaledContents(True)

    @staticmethod
    def pil_to_qimage(pil_image):
       
        pil_image = pil_image.convert("RGBA")
        data = pil_image.tobytes("raw", "RGBA")
        q_image = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
        return q_image

    
