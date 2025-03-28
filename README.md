# Image Processing Task - README  

## Overview  
This project involves applying various image processing techniques to standard grayscale and color images. The main goal is to manipulate, filter, and analyze images using different noise models, filtering techniques, edge detection methods, histogram analysis, and frequency domain processing.  

## Tasks to Implement  

### **1. Adding Additive Noise**  
Introduce noise into the image using the following types:  
- **Uniform Noise**  
- **Gaussian Noise**  
- **Salt & Pepper Noise**  

### **2. Filtering Noisy Images**  
Apply the following low-pass filters to reduce noise:  
- **Average Filter**  
- **Gaussian Filter**  
- **Median Filter**  

### **3. Edge Detection**  
Detect edges using the following masks:  
- **Sobel Operator**  
- **Roberts Operator**  
- **Prewitt Operator**  
- **Canny Edge Detector**  

### **4. Histogram and Distribution Curve**  
- Compute and display the image histogram.  
- Plot the distribution function (cumulative histogram).  

### **5. Histogram Equalization**  
- Apply histogram equalization to enhance image contrast.  

### **6. Image Normalization**  
- Normalize pixel intensity values for improved contrast.  

### **7. Local and Global Thresholding**  
- Apply different thresholding techniques to segment the image.  

### **8. Color-to-Grayscale Transformation**  
- Convert a color image to grayscale.  
- Plot histograms for **Red (R), Green (G), and Blue (B)** channels.  
- Compute the cumulative distribution function for mapping and histogram equalization.  

### **9. Frequency Domain Filtering**  
- Implement **low-pass** and **high-pass** frequency domain filters.  

### **10. Hybrid Images**  
- Create hybrid images by combining different frequency components from two images.    

## **Setup & Dependencies**  
Ensure you have the following Python libraries installed:  
```bash
pip install numpy opencv-python matplotlib scipy
```
Or, if using Jupyter Notebook, include:  
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
```

## **How to Run the Code**  
1. Load an input image (grayscale or color).  
2. Run the scripts for each task as needed.  
3. Save and analyze results.  
