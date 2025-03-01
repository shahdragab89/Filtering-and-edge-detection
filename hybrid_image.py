import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from frequencyfilters import FrequencyFilters
from scipy.ndimage import zoom

class HybridImageProcessor:
    def __init__(self, ui):
        
        self.ui = ui
        self.original_image1 = None
        self.original_image2 = None
        
        # Connect UI elements
        self.ui.picture_slider.valueChanged.connect(self.update_images)
        self.ui.picture_slider_2.valueChanged.connect(self.update_images)
    
    def set_original_images(self, image1, image2):
       
        self.original_image1 = image1
        self.original_image2 = image2
        self.update_images()
    
    def resize_image(self, image, target_shape):
        
        scale_factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
        return zoom(image, scale_factors, order=1)  # Bilinear interpolation
    
    def update_images(self):
       
        if self.original_image1 is None and self.original_image2 is None:
            return

        target_shape = None
        if self.original_image1 is not None and self.original_image2 is not None:
            target_shape = (min(self.original_image1.shape[0], self.original_image2.shape[0]),
                            min(self.original_image1.shape[1], self.original_image2.shape[1]))
        elif self.original_image1 is not None:
            target_shape = self.original_image1.shape
        elif self.original_image2 is not None:
            target_shape = self.original_image2.shape

        # Process Image 1 (If exists, apply low-pass filter)
        if self.original_image1 is not None:
            image1 = self.resize_image(self.original_image1, target_shape)
            radius1 = self.ui.picture_slider.value()
            filtered_image1 = FrequencyFilters.ideal_low_pass_filter(image1, radius1)
            filtered_qimage1 = FrequencyFilters.convert_to_qimage(filtered_image1)
            self.ui.image1.setPixmap(QPixmap.fromImage(filtered_qimage1))
            self.ui.image1.setScaledContents(True)
        else:
            self.ui.image1.clear()  

        # Process Image 2 (If exists, apply high-pass filter)
        if self.original_image2 is not None:
            image2 = self.resize_image(self.original_image2, target_shape)
            radius2 = self.ui.picture_slider_2.value()
            filtered_image2 = FrequencyFilters.ideal_high_pass_filter(image2, radius2)
            filtered_qimage2 = FrequencyFilters.convert_to_qimage(filtered_image2)
            self.ui.image2.setPixmap(QPixmap.fromImage(filtered_qimage2))
            self.ui.image2.setScaledContents(True)
        else:
            self.ui.image2.clear()  

        if self.original_image1 is not None and self.original_image2 is not None:
            hybrid_image = np.clip(filtered_image1.astype(np.float32) + filtered_image2.astype(np.float32), 0, 255).astype(np.uint8)
            hybrid_qimage = FrequencyFilters.convert_to_qimage(hybrid_image)
            self.ui.hyprid_image.setPixmap(QPixmap.fromImage(hybrid_qimage))
            self.ui.hyprid_image.setScaledContents(True)
        else:
            self.ui.hyprid_image.clear()  

        print("Filtered images and hybrid image updated")
