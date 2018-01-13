import cv2
import numpy as np

from packages.image_color_transform import ColorTransformer
from packages.image_gradient import Gradient


class LaneDetectorPreprocessor(object):
    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))

    def preprocess(self, img):
        # Define the output image
        binary_mask = np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.uint8)

        # Pre-Filtering with gaussian smoothing on the RGB-Colorspace image
        img = cv2.blur(img, ksize=(3, 3))

        # Extract the S - Channel from HLS-Colorspace of this frame
        img_s = np.expand_dims(ColorTransformer.transformBGR2HLS(img)[:, :, 2], axis=-1)

        # Apply Contrast Limited Adaptive Histogram Equalization
        img_s[:, :, 0] = self._clahe.apply(img_s)

        # Compute Derivatives with Sobel-Kernels in x and y
        sobel_x, sobel_y, mag, direction = Gradient.get_gradient_images(img_s, sobel_kernel_size=5)

        # Initialize 'Conditions' image
        conditions_combination = np.zeros(shape=(img_s.shape[0], img_s.shape[1], 3), dtype=np.uint8)

        # Create condition mask / map for gradient direction and gradient magnitude
        condition_grad = (direction >= 0.0) & (direction <= 0.78)
        condition_grad &= (mag >= 15) & (mag <= 255)

        # Create condition mask / map for intensity in the S-Channel of HLS-Colorspace
        condition_color = (img_s[:, :, 0] >= 200) & (img_s[:, :, 0] <= 255)

        # Create a combination for better visualisation in different color channels (BGR)
        conditions_combination[:, :, 0] = condition_grad * 255
        conditions_combination[:, :, 1] = condition_color * 255

        # Finally use the condition masks to create the final binary mask (0, 255)
        binary_mask[condition_color | condition_grad] = 255

        # Filter with morphological operations (open)
        binary_mask = cv2.erode(binary_mask, np.ones((2, 2), np.uint8), 1)
        binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))

        return binary_mask, img_s, mag, conditions_combination
