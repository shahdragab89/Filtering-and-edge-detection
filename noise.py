import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig
from PyQt5.QtGui import QPixmap, QImage




class NoiseProcessor:
    @staticmethod
    def add_salt_and_pepper_noise(image, prob = 0.05):
        noisy_image = np.zeros(image.shape,np.uint8)
        thres = 1 - prob
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

    def add_gaussian_noise(image, mean=0, std=25):
        image = image.astype(np.float32)
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image
    
    def add_uniform_noise(image, intensity=30):
        image = image.astype(np.float32)
        noise = np.random.uniform(-intensity, intensity, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image


    def applyNoiseAndDisplay(noiseType, image):
        print("apply noise")
        if noiseType== "Salt & Pepper":
            noisy_image = NoiseProcessor.add_salt_and_pepper_noise(image)
        elif noiseType == "Gaussian":
            noisy_image = NoiseProcessor.add_gaussian_noise(image)
        elif noiseType == "Uniform":
            noisy_image = NoiseProcessor.add_uniform_noise(image)

        # Convert to QImage
        height, width, channel = noisy_image.shape
        bytes_per_line = 3 * width
        noisy_image = QImage(noisy_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        return noisy_image