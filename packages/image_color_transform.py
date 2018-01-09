"""
This module is for transforming an image by color format.
It is just a tiny wrapper for opencv cvtColor
"""
import cv2


class ColorTransformer(object):
    def __init(self):
        pass

    @staticmethod
    def transformBGR2HLS(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    @staticmethod
    def transform_BGR2GRAY(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
