import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QBuffer, QIODevice
import PIL.ImageQt as ImageQtModule

# Manually patch QBuffer and QIODevice into ImageQt
ImageQtModule.QBuffer = QBuffer
ImageQtModule.QIODevice = QIODevice



class NoiseProcessor:
    @staticmethod
    def add_salt_and_pepper_noise(image, prob = 0.05):
        noisy_image = image.copy() 
        # Threshold to determine white noise
        thres = 1 - prob
        # Iterate through each pixel and apply noise
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    noisy_image[i][j] = 0
                elif rdn > thres:
                    noisy_image[i][j] = 255
                else:
                    noisy_image[i][j] = image[i][j]
        return noisy_image

    @staticmethod
    def add_gaussian_noise(image, meanValue=0, std=25):
        image = image.astype(np.float32)
        # Generate Gaussian noise
        noise = np.random.normal(meanValue, std, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    @staticmethod
    def add_uniform_noise(image, intensity=30):
        image = image.astype(np.float32)
        # Generate uniform noise
        noise = np.random.uniform(-intensity, intensity, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def applyNoiseAndDisplay(noiseType, image, meanValue=0):
        print("apply noise")
        if noiseType== "Salt & Pepper":
            noisy_image = NoiseProcessor.add_salt_and_pepper_noise(image)
        elif noiseType == "Gaussian":
            noisy_image = NoiseProcessor.add_gaussian_noise(image, meanValue)
        elif noiseType == "Uniform":
            noisy_image = NoiseProcessor.add_uniform_noise(image)

        # Convert to QImage
        height, width = noisy_image.shape
        bytes_per_line = width

        NoiseProcessor.last_noisy_image = noisy_image
        noisy_image = QImage(noisy_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        return noisy_image