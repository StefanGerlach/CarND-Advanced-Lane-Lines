""" This module is for transforming an image by a perspective transform. """

import numpy as np
import pickle
import cv2
import os


class PerspectiveTransform(object):
    def __init__(self):
        # Initialize transformation matrix with identity matrix for no change.
        self._matrix = np.identity(3)
        self._inv_matrix = np.identity(3)

    def save(self, fn: str):
        """
        Saves the matrix to file.
        """
        pickle.dump(dict({'transform': self._matrix, 'inverse': self._inv_matrix}), open(fn, 'wb'))

    def load(self, fn: str):
        """
        This function loads a pickle file that contains the key 'transform' with the
        perspective transformation.
        :param fn: path to file
        :return: None
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError('Could not find file '+fn+' !')

        pickle_dict = pickle.load(open(fn, 'rb'))
        if 'transform' not in pickle_dict or 'inverse' not in pickle_dict:
            raise ValueError('Could not find key <transform> or <inverse> in pickle file')

        self._matrix = pickle_dict['transform']
        self._inv_matrix = pickle_dict['inverse']

    def from_points_pair(self, pts_src, pts_dst):
        """
        Uses cv2.getPerspectiveTransform to calculate perspective transform from 4 point pairs!
        :param pts_src: 4 Points in image A
        :param pts_dst: 4 Points in image B
        """
        if len(pts_src) != 4 or len(pts_dst) != 4:
            raise ValueError('Needs exactly 4 points in list.')

        self._matrix = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
        self._inv_matrix = cv2.getPerspectiveTransform(np.float32(pts_dst), np.float32(pts_src))

    def apply(self, img, dst_size=None):
        """
        Returns the input image transformed by the 'self'-transformation matrix
        :param img: input image
        :return: transformed image.
        """
        if dst_size is None:
            dsize = (img.shape[1], img.shape[0])
        else:
            dsize = dst_size

        return cv2.warpPerspective(img, self._matrix, dsize)

    def apply_inv(self, img, dst_size=None):
        """
        Returns the input image transformed by the inverse 'self'-transformation matrix
        :param img: input image
        :return: transformed image.
        """
        if dst_size is None:
            dsize = (img.shape[1], img.shape[0])
        else:
            dsize = dst_size

        return cv2.warpPerspective(img, self._inv_matrix, dsize)


"""
# TEST CODE
t = PerspectiveTransform()
imgA = cv2.imread('test.jpg')
t.from_points_pair([(0,0), (270,0), (270, 240), (0, 240)], [(0,0), (1280, 0), (1280, 720), (0, 720)])
t.save('perspective_transform.p')

t2 = PerspectiveTransform()
t2.load('perspective_transform.p')
imgB = t2.apply(imgA)
cv2.imwrite('test3.jpg', imgB)
"""




