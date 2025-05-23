import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox

class ImageNormalizer:
    def __init__(self, label):
        """
        Initialize the ImageNormalizer with a QLabel where the normalized image will be displayed.
        :param label: QLabel where the normalized image will be shown.
        """
        self.label = label  # QLabel where the image will be displayed
        self.loaded_image = None  # Store the uploaded image

    def set_image(self, image):
        """ Set the uploaded image to be normalized later. """
        self.loaded_image = image

    def normalize_image_and_display(self):
        """ Normalize the image and display it in the provided QLabel. """
        if self.loaded_image is None:
            QMessageBox.warning(None, "Warning", "Please load an image before normalizing.")
            return

        try:
            # Normalize the image
            normalized_image = self.normalize_image(self.loaded_image)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"An error occurred during normalization: {str(e)}")
            return

        if normalized_image is not None:
            # Convert normalized image to QImage for display
            height, width = normalized_image.shape[:2]

            if len(normalized_image.shape) == 2:  # Grayscale
                bytes_per_line = width  # 1 byte per pixel
                normalized_qimage = QImage(normalized_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # RGB
                bytes_per_line = width * 3
                normalized_qimage = QImage(normalized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            normalized_pixmap = QPixmap.fromImage(normalized_qimage)

            # Display the normalized image on the QLabel
            self.label.setPixmap(normalized_pixmap.scaled(self.label.size()))

    def normalize_image(self, image):
        """ Convert QImage to NumPy array and normalize it. """

        # Convert QImage to grayscale NumPy array
        if image.format() in [QImage.Format_RGB32, QImage.Format_RGB888, QImage.Format_RGBA8888]:
            image = image.convertToFormat(QImage.Format_Grayscale8)  # Convert to grayscale if not already

        # Convert QImage to NumPy array
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape(image.height(), image.width())

        # Normalize using OpenCV's min-max normalization
        normalized_image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return normalized_image
