import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class FrequencyFilters:
    @staticmethod
    def apply_dft(image):
        """Computes the Discrete Fourier Transform (DFT) of the image."""
        image = image.astype(np.float32)  # Convert to 32-bit floating point
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)  # Center the low frequencies
        return dft_shift

    @staticmethod
    def apply_idft(dft_shift):
        """Computes the Inverse Discrete Fourier Transform (IDFT) to get back the image."""
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_back = np.fft.ifft2(dft_ishift)
        img_back = np.abs(img_back)  # Convert to real values
        return img_back.astype(np.uint8)

    @staticmethod
    def ideal_low_pass_filter(image, radius):
        """Applies an Ideal Low-Pass Filter (ILPF) in the frequency domain."""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates
        
        # Create a mask with a centered circle (low-pass filter)
        mask = np.zeros((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius**2
        mask[mask_area] = 1

        # Apply the mask
        dft_shift = FrequencyFilters.apply_dft(image)
        dft_shift_filtered = dft_shift * mask

        # Convert back to the spatial domain
        return FrequencyFilters.apply_idft(dft_shift_filtered)

    @staticmethod
    def ideal_high_pass_filter(image, radius):
        """Applies an Ideal High-Pass Filter (IHPF) in the frequency domain."""
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask with a centered circle (high-pass filter)
        mask = np.ones((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius**2
        mask[mask_area] = 0

        # Apply the mask
        dft_shift = FrequencyFilters.apply_dft(image)
        dft_shift_filtered = dft_shift * mask

        # Convert back to the spatial domain
        return FrequencyFilters.apply_idft(dft_shift_filtered)
    
    @staticmethod
    def apply_frequency_filter(image, filter_type, radius):
        """Applies the selected frequency domain filter to the image."""
        if filter_type == "Ideal Low":
            return FrequencyFilters.ideal_low_pass_filter(image, radius)
        elif filter_type == "Ideal High":
            return FrequencyFilters.ideal_high_pass_filter(image, radius)
        else:
            raise ValueError("Invalid frequency filter type. Choose 'Ideal Low' or 'Ideal High'.")

    @staticmethod
    def convert_to_qimage(image):
        """Converts a NumPy grayscale image to QImage for displaying in PyQt."""
        image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure the image is uint8
        height, width = image.shape
        bytes_per_line = width
        return QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)

    @staticmethod
    def process_and_display(image, filter_type, radius):
        """Applies frequency filter and converts to QImage."""
        filtered_image = FrequencyFilters.apply_frequency_filter(image, filter_type, radius)
        return FrequencyFilters.convert_to_qimage(filtered_image)
    
    @staticmethod
    def handle_frequency_filter(image, filter_type, radius):
        """Handles the application of a frequency domain filter and returns QImage for display."""
        try:
            if image is None:
                raise ValueError("No image loaded. Please upload an image before applying frequency domain filters.")
            return FrequencyFilters.process_and_display(image, filter_type, radius)
        except ValueError as ve:
            print(f"Error: {ve}")
            return None
    
    @staticmethod
    def update_frequency_filter(ui):
        """Updates the displayed image based on the selected frequency filter and radius."""
        filter_type = ui.freqFilter_comboBox.currentText()
        radius = ui.radius_slider_2.value()
        
        if ui.image is None:
            print("Error: No image loaded. Please upload an image before applying frequency domain filters.")
            return

        filtered_qimage = FrequencyFilters.handle_frequency_filter(ui.image, filter_type, radius)
        
        if filtered_qimage:
            ui.filtered_image.setPixmap(QPixmap.fromImage(filtered_qimage))
            ui.filtered_image.setScaledContents(True)

    @staticmethod
    def connect_ui_elements(ui):
        """Connects UI elements to frequency filter update function."""
        ui.freqFilter_comboBox.activated.connect(lambda: FrequencyFilters.update_frequency_filter(ui))
        ui.radius_slider_2.valueChanged.connect(lambda: FrequencyFilters.update_frequency_filter(ui))
