from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
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
        gray_image = self.to_grayscale(image_array)

        # Local Thresholding
        block_size = 11  
        constant = 2     
        local_thresh = self.local_thresholding(gray_image, block_size, constant)
        self.display_thresholded_image(local_thresh, self.label_local)

        # Global Thresholding
        global_thresh = self.global_thresholding(gray_image)
        self.display_thresholded_image(global_thresh, self.label_global)

    def qimage_to_numpy(self, qimage):
        image_format = QImage.Format_RGB888 if qimage.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        qimage = qimage.convertToFormat(image_format)
        channels = 3 if image_format == QImage.Format_RGB888 else 4
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        return np.array(ptr).reshape(qimage.height(), qimage.width(), channels)

    def to_grayscale(self, image):
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def global_thresholding(self, image):
        threshold = 128  # Fixed threshold
        return np.where(image > threshold, 255, 0).astype(np.uint8)


    def local_thresholding(self, image, block_size, constant):
        pad_size = block_size // 2
        padded_image = np.pad(image, pad_size, mode='reflect')
        thresholded = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_region = padded_image[i:i+block_size, j:j+block_size]
                local_thresh = np.mean(local_region) - constant
                thresholded[i, j] = 255 if image[i, j] > local_thresh else 0

        return thresholded

    def display_thresholded_image(self, thresholded_image, label):
        height, width = thresholded_image.shape
        bytes_per_line = width
        q_image = QImage(thresholded_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        thresholded_pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(thresholded_pixmap.scaled(label.size()))
