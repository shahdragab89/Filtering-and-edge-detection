from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
import cv2 as cv
import numpy as np

class ImageThresholding:
    def __init__(self, label_local, label_global):
        self.label_local = label_local
        self.label_global = label_global
        self.loaded_image = None

    def set_image(self, image):
        self.loaded_image = image
    
    def apply_thresholding(self):
        if self.loaded_image is None:
            QMessageBox.warning(None, "Warning", "Please load an image before thresholding.")
            return

        image_array = self.qimage_to_numpy(self.loaded_image)
        gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

        # **Local Thresholding**
        block_size = 11  
        constant = 2     
        local_thresh = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, constant)
        self.display_thresholded_image(local_thresh, self.label_local)

        # **Global Thresholding**
        _, global_thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        self.display_thresholded_image(global_thresh, self.label_global)
    
    def qimage_to_numpy(self, qimage):
        image_format = QImage.Format_RGB888 if qimage.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        qimage = qimage.convertToFormat(image_format)
        channels = 3 if image_format == QImage.Format_RGB888 else 4
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        return np.array(ptr).reshape(qimage.height(), qimage.width(), channels)

    def display_thresholded_image(self, thresholded_image, label):
        if len(thresholded_image.shape) == 3:
            thresholded_image = cv.cvtColor(thresholded_image, cv.COLOR_BGR2GRAY)

        height, width = thresholded_image.shape
        bytes_per_line = width
        q_image = QImage(thresholded_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        thresholded_pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(thresholded_pixmap.scaled(label.size()))
