"""
This package implements the code for camera calibration.
The main part of this module is from my fork of the Udacity CarND-Camera-Calibration repository.

https://github.com/StefanGerlach/CarND-Camera-Calibration
"""

import cv2
import os
import glob
import numpy as np
import pickle           # for saving the camera calibration metices


class CameraCalibration(object):
    def __init__(self, camera_id: str):
        self._chessboard_shape = (9, 6)
        self._cal_matrix_fn = 'cam_cal_' + camera_id + '.p'

    def save_calibration(self, calibration_dict: dict, dst_path: str):
        """
        Saves a camera calibration dictionary with mtx and dist.
        The filename is defined by the camera_id that was given in constructor.
        Creates directory if it does not exist.
        :param calibration_dict: Dictionary with 'mtx' and 'dist' entries.
        :param dst_path: Directory where to save the pickle file.
        :return: None
        """
        if os.path.exists(dst_path) is False:
            os.makedirs(dst_path)

        pickle.dump(calibration_dict, open(os.path.join(dst_path, self._cal_matrix_fn), 'wb'))

    def load_calibration(self, path: str):
        """
        This function is for reading camera calibration metrices that have been calculated before.
        :param path: Path to directory in which the file is located.
        :return: True/False, Dict({'mtx', 'dist'})
        """
        if not os.path.isdir(path):
            return False, None

        filename = os.path.join(path, self._cal_matrix_fn)
        if not os.path.isfile(filename):
            return False, None

        return True, pickle.load(open(filename, 'rb'))

    def read_images_gen_cal(self, path: str) -> dict:
        """
        This function takes a path and looks for png and jpg files in the given location.
        The images are loaded and used for camera calibration matrix calculation.
        :param path: Directory with images for camera calibration.
        :return: Dictionary with 'mtx' and 'dist' as components of the camera calibration.
        """
        image_files = []
        # read in all image paths (for png and jpgs)
        image_files.extend(glob.glob(os.path.join(path, '*.png')))
        image_files.extend(glob.glob(os.path.join(path, '*.jpg')))

        if len(image_files) <= 0:
            raise FileNotFoundError('Could not find any image files for camera calibration (png, jppgs supported!')

        # preparation of the 'real world' points like in CarND-Camera-Calibration repository ipynb!
        syn_objp = np.zeros((self._chessboard_shape[1] * self._chessboard_shape[0], 3), np.float32)
        syn_objp[:, :2] = np.mgrid[0:self._chessboard_shape[0], 0:self._chessboard_shape[1]].T.reshape(-1, 2)

        # Arrays to store all points pairs from the images. They will be used for calibration computation.
        objpoints = []
        imgpoints = []
        img_size = None

        for file_name in image_files:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # remember shape of images
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            # Chessboard detection
            res, chess_pts = cv2.findChessboardCorners(gray, self._chessboard_shape, None)
            if res:
                objpoints.append(syn_objp)
                imgpoints.append(chess_pts)

        if len(objpoints) <= 0:
            raise ValueError('No Chessboard detected on images.')

        # Do camera calibration given object points and image points
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        if not ret:
            raise ValueError('cv2.calibrateCamera() failed.')

        # prepare return value dictionary
        ret = {'mtx': mtx, 'dist': dist}
        return ret

    def undistort_image(self, img, calibration_dict: dict):
        """
        This function uses the values of a camera calibration for undistort an image.
        :param img: Image that should be undistorted.
        :param calibration_dict: Dictionary with keys 'mtx' and 'dist'.
        :return: Undistorted image.
        """
        if img is None:
            return None

        if 'mtx' not in calibration_dict or 'dist' not in calibration_dict:
            raise ValueError('Missing mtx or dist in calibration dictionary.')

        return cv2.undistort(img, calibration_dict['mtx'], calibration_dict['dist'], None, calibration_dict['mtx'])
