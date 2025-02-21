import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig
from PyQt5.QtGui import QPixmap, QImage
from scipy.ndimage import convolve 

class FilterProcessor:
    @staticmethod


    def average_filter(image, maskSize=(3, 3)):
        if isinstance(maskSize, int):
            maskSize = (maskSize, maskSize)
        mask = np.ones(maskSize, dtype=np.float32) / (maskSize[0] * maskSize[1])
        rows, cols = image.shape
        filter_height, filter_width = mask.shape
        pad_h = filter_height // 2
        pad_w = filter_width // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i + filter_height, j:j + filter_width]
                filtered_image[i, j] = np.sum(region * mask)

        return filtered_image.astype(np.uint8)  

        

    def gaussian_filter(image, mask_size=3, sigma=1):
        # Ensure mask_size is odd
        if mask_size % 2 == 0:
            raise ValueError("mask_size must be an odd number.")

        # Create Gaussian kernel
        ax = np.arange(-(mask_size // 2), (mask_size // 2) + 1)
        x, y = np.meshgrid(ax, ax)
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian /= np.sum(gaussian)  
        filtered_image = convolve(image, gaussian, mode='constant', cval=0.0)
        
        return filtered_image.astype(np.uint8)

        

    def median_filter(image, filter_size=3):
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be an odd number.")

        # Pad the image with zeros to handle edge cases
        pad_size = filter_size // 2
        padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
        filtered_image = np.zeros_like(image, dtype=np.uint8)
        rows, cols = image.shape

        # Apply median filter
        for i in range(rows):
            for j in range(cols):
                region = padded_image[i:i+filter_size, j:j+filter_size]
                filtered_image[i, j] = np.median(region)

        return filtered_image

        

    def applyFilterAndDisplay(filterType, sliderValues, image):
        print("apply filter")
        if filterType == "Average": 
            filtered_image = FilterProcessor.average_filter(image, sliderValues[0])
        elif filterType == "Gaussian":
            filtered_image = FilterProcessor.gaussian_filter(image, sliderValues[0], sliderValues[1])
        elif filterType == "Median": 
            filtered_image = FilterProcessor.median_filter(image)

        # Convert to QImage
        height, width = filtered_image.shape
        bytes_per_line = width
        filtered_image = QImage(filtered_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return filtered_image
        