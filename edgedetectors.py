import cv2
import numpy as np
from PyQt5.QtGui import QImage
from scipy.ndimage import gaussian_filter


class EdgeDetector:
    
    @staticmethod
    def apply_edge_detection(image, method, sigma = 1):
        if image is None:
            raise ValueError("No image provided for edge detection")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected an image as a NumPy array")
        
        # convert image to grayscale ((if not already))
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
        
        # normalize to 0-255 if needed
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        result = None
        
        if method.lower() == 'prewitt':
            result = EdgeDetector.prewitt(img)
        elif method.lower() == 'roberts':
            result = EdgeDetector.roberts(img)
        elif method.lower() == 'sobel':
            result = EdgeDetector.sobel(img)
        elif method.lower() == 'canny':
            result = EdgeDetector.canny(img, sigma)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # conversion to QImage 
        height, width = result.shape
        bytes_per_line = width
        return QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    @staticmethod
    def prewitt(image):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude

    @staticmethod
    def roberts(image):
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude

    @staticmethod
    def sobel(image):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)

        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude

    @staticmethod
    def canny(image, sigma):
        #1st step: gaussian blur 
        smoothed = gaussian_filter(image, sigma=sigma)
        
        #2nd step: grad mag & direction
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = cv2.filter2D(smoothed, cv2.CV_32F, sobel_x)
        grad_y = cv2.filter2D(smoothed, cv2.CV_32F, sobel_y)

        magnitude = np.hypot(grad_x, grad_y)
        magnitude = (magnitude / magnitude.max()) * 255 
        direction = np.arctan2(grad_y, grad_x) 
        
        #3rd step: non-max supression
        nms = np.zeros_like(magnitude, dtype=np.uint8)
        angle = direction * (180 / np.pi)  
        angle[angle < 0] += 180  
        
        rows, cols = magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q, r = 255, 255
                #neighbors based on grad direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q, r = magnitude[i, j + 1], magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q, r = magnitude[i + 1, j], magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
                
                #suppress non-max values
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    nms[i, j] = magnitude[i, j]
                else:
                    nms[i, j] = 0
        
        #4th step: hysterisis thresholding 
        high_thresh = np.max(nms) * 0.2 * (1 + sigma / 10)
        low_thresh = high_thresh * 0.5
        
        strong_edges = (nms >= high_thresh).astype(np.uint8) * 255
        weak_edges = ((nms >= low_thresh) & (nms < high_thresh)).astype(np.uint8) * 50
        
        #connect weak edges to strong edges
        final_edges = strong_edges.copy()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if weak_edges[i, j] == 50:
                    if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                        final_edges[i, j] = 255
        
        return final_edges