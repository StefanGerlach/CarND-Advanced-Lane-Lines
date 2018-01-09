"""
This package implements mostly the parts of the examples from the course material.
"""
import cv2
import numpy as np


class Gradient(object):

    @staticmethod
    def get_gradient_images(img, sobel_kernel_size=7):
        """
        This function takes an BGR image and computes the derivatives in X and Y with Sobel-Operator.
        Additionally it computes the magnitude and direction of the gradient, normed to 0..255.
        :param image: input BGR color image.
        :return: sobel_x, sobel_y, magnitude, direction
        """
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # 2) Take the gradient in x and y separately
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel_size)

        # 3) Calculate the magnitude
        mag = np.sqrt(np.add(np.square(sobel_x), np.square(sobel_y)))

        # 4) Calculate the direction
        directions = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

        # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        mag = np.uint8(255 * mag / np.max(mag))

        sobel_x = np.uint8(255 * np.absolute(sobel_x) / np.max(sobel_x))
        sobel_y = np.uint8(255 * np.absolute(sobel_y) / np.max(sobel_y))

        return sobel_x, sobel_y, mag, directions

