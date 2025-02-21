import random
import statistics as stat
import cv2
import numpy as np
import scipy.signal as sig
from PyQt5.QtGui import QPixmap, QImage




class NoiseProcessor:
    @staticmethod
    # def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    #     print("salt and pepper")
    #     """
    #     Adds salt-and-pepper noise to the given image.
        
    #     :param image: Input image (numpy array).
    #     :param salt_prob: Probability of adding salt noise (white pixels).
    #     :param pepper_prob: Probability of adding pepper noise (black pixels).
    #     :return: Noisy image (numpy array).
    #     """
    #     noisy_image = image.copy()
    #     height, width, _ = image.shape
        
    #     # Add salt (white pixels)
    #     num_salt = int(salt_prob * height * width)
    #     salt_coords = [np.random.randint(0, i, num_salt) for i in (height, width)]
    #     noisy_image[salt_coords[0], salt_coords[1]] = [255, 255, 255]  # White pixels

    #     # Add pepper (black pixels)
    #     num_pepper = int(pepper_prob * height * width)
    #     pepper_coords = [np.random.randint(0, i, num_pepper) for i in (height, width)]
    #     noisy_image[pepper_coords[0], pepper_coords[1]] = [0, 0, 0]  # Black pixels

    #     return noisy_image
    def add_salt_and_pepper_noise(image, prob = 0.05):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
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

    # def add_gaussian_noise(image, mean=0, std=0.1):
    #     noise = np.multiply(np.random.normal(mean, std, image.shape), 255)
    #     noisy_image = np.clip(image.astype(int)+noise, 0, 255)
    #     return noisy_image

    # def add_uniform_noise(image, prob=0.1):
    #     levels = int((prob * 255) // 2)
    #     noise = np.random.uniform(-levels, levels, image.shape)
    #     noisy_image = np.clip(image.astype(int) + noise, 0, 255)
    #     return noisy_image
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
        """
        Applies salt-and-pepper noise to the uploaded image and displays it in filtered_image QLabel.
        """
        print("apply noise")
        
        # if hasattr(self, 'cv_image') and self.cv_image is not None:
            
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