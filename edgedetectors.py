import cv2
import numpy as np
from PyQt5.QtGui import QImage


class EdgeDetector:
    
    @staticmethod
    def apply_edge_detection(image, method, threshold1=50, threshold2=150, aperture_size=3):
        if image is None:
            raise ValueError("No image provided for edge detection")
        
        # Ensure the input is a NumPy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected an image as a NumPy array")
        
        # Convert image to grayscale if not already
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image.copy()
        
        # Normalize to 0-255 if needed
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
            result = EdgeDetector.canny(img, threshold1, threshold2, aperture_size)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Convert to QImage for display
        height, width = result.shape
        bytes_per_line = width
        return QImage(result.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    
    @staticmethod
    def prewitt(image):
        """Apply Prewitt edge detection"""
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude
    
    @staticmethod
    def roberts(image):
        """Apply Roberts edge detection"""
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude
    
    @staticmethod
    def canny(image, threshold1=50, threshold2=150, aperture_size=3):
        """Apply Canny edge detection"""
        edges = cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)
        return edges
    
    @staticmethod
    def sobel(image):
        """Apply 3x3 Sobel edge detection using kernels"""
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        grad_x = cv2.filter2D(image, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_32F, kernel_y)

        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return magnitude

    
