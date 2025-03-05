import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtWidgets import QLabel
from PyQt5 import Qt
from io import BytesIO  
from histogram import calc_hist

class Plot_CDF:
    def __init__(self, image, label: QLabel):
        if image is None or isinstance(image, bool): 
            print("Error: Invalid image provided for CDF computation.")
            # Clear the label to ensure nothing is displayed
            label.clear()
            return
        
        self.image = image 
        self.label = label  
        self.plot_cdf_to_label()

    def plot_cdf_to_label(self):
        if self.image is None:
            print("No valid image loaded for CDF computation.")
            self.label.clear()
            return
        
        try:
            # Get label dimensions
            label_width = self.label.width()
            label_height = self.label.height()

            plt.figure(figsize=(5, 4), dpi=100)
            
            # Calculate histogram
            hist = calc_hist([self.image], [0], None, [256], [0, 256])
            
            # Calculate PDF and CDF
            pdf = hist / hist.sum()
            cdf = np.cumsum(pdf)
            
            # Plot CDF
            plt.plot(cdf, color='black')
            plt.title("Grayscale CDF Distribution")
            plt.xlabel("Pixel Value")
            plt.ylabel("Cumulative Probability")
            plt.xlim([0, 256])
            plt.ylim([0, 1.05])
            plt.grid()
            
            # Convert plot to QPixmap
            fig = plt.gcf()
            fig.canvas.draw()
            
            # Convert the plot to a QImage
            buf = fig.canvas.buffer_rgba()
            ncols, nrows = fig.canvas.get_width_height()
            q_image = QImage(buf, ncols, nrows, QImage.Format_RGBA8888)
            
            # Create QPixmap and scale it
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to 90% of label size while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                int(label_width), 
                int(label_height), 
                Qt.Qt.KeepAspectRatio, 
                Qt.Qt.SmoothTransformation
            )
            
            # Set the scaled pixmap to the label
            self.label.setPixmap(scaled_pixmap)
            
            # Center the pixmap in the label
            self.label.setAlignment(Qt.Qt.AlignCenter)
            
            # Close the figure to free up memory
            plt.close(fig)
        
        except Exception as e:
            print(f"Error plotting CDF: {e}")
            self.label.clear()