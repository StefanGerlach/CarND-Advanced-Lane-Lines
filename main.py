"""
In main.py the complete pipeline will be included.
Here the single modules will be called that do:

 - Camera Calibration
 - Perspective Image Transformation for Bird-Eye-View
 - Image Color Conversion and Channel Extraction
 - Image Gradient Calculation
 - Combined Thresholding techniques
 - Lane Detection
 - Filtering
 - Visualisation

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from packages.camera_calibration import CameraCalibration
from packages.image_transform import PerspectiveTransform
from packages.image_color_transform import ColorTransformer
from packages.image_gradient import Gradient


""" Definition of some globals """

# For intermediate outputs
output_path = 'output_images'

# For input of camera calibration
cal_path = 'camera_cal'
camera_id = 'udacity_video_camera'

# Directory for image transformations
transform_path = 'image_transform'

# Eye-balled position for perspective transformation
transform_pts_a = [(565, 478), (740, 478), (1095, 675), (315, 675)]
transform_pts_b = [(128, 128), (384, 128), (384, 512), (128, 512)]
transform_dst_size = (512, 512)

# Video files
# video_file = 'project_video.mp4'
video_file = 'challenge_video.mp4'
# video_file = 'harder_challenge_video.mp4'

"""                                         """
""" Part 1 Camera Calibration Preparation   """
"""                                         """

# Create Instance of CameraCalibration class
camera_cal = CameraCalibration(camera_id=camera_id)

# Try to load a previously calculated matrix.
ret, cal = camera_cal.load_calibration(path=cal_path)
if not ret:
    # If there was no camera calibration data, create and save it!
    cal = camera_cal.read_images_gen_cal(path=cal_path)
    camera_cal.save_calibration(calibration_dict=cal, dst_path=cal_path)

# Test and visualize a single image
img = cv2.imread(os.path.join(cal_path, 'calibration1.jpg'))
img_undistort = camera_cal.undistort_image(img, calibration_dict=cal)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(img_undistort)
ax2.set_title('Undistorted Image', fontsize=30)
plt.savefig(os.path.join(output_path, 'camera_cal_test.png'))


"""                                                     """
""" Part 2 Perspective Image Transformation Preparation """
"""                                                     """

# Create the Transformer
transformer = PerspectiveTransform()

# Check the Directory for an image transformation
if not os.path.isdir(transform_path):
    os.makedirs(transform_path)

transform_filename = os.path.join(transform_path, 'transform.p')

if not os.path.isfile(transform_filename):
    # There is no Bird-Eye Transform, so we have to create it.
    transformer.from_points_pair(transform_pts_a, transform_pts_b)
    transformer.save(transform_filename)

# Load transformation
transformer.load(transform_filename)


"""                             """
""" Part 3 Main Pipeline        """
"""                             """

# Playing a video from file is very well described in the opencv documentation:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

# Open video file
clip = cv2.VideoCapture(video_file)

# Preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))

# Iterate all frames of the video
frame_id = 0
while clip.isOpened():
    _, frame = clip.read()

    # Undistort the camera radial distortion
    frame = camera_cal.undistort_image(frame, calibration_dict=cal)
    frame = transformer.apply(frame, transform_dst_size)
    frame = cv2.blur(frame, ksize=(5, 5))

    frame_clone = frame.copy()
    #frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    binary_mask = np.zeros(shape=(frame.shape[0], frame.shape[1], 1), dtype=np.uint8)

    # Extract the S - Channel from HLS-Colorspace of this frame
    frame_S = np.expand_dims(ColorTransformer.transformBGR2HLS(frame)[:, :, 2], axis=-1)
    frame_S[:, :, 0] = clahe.apply(frame_S)

    frame_GRAY = np.expand_dims(ColorTransformer.transform_BGR2GRAY(frame), axis=-1)

    sobel_x, sobel_y, mag, direction = Gradient.get_gradient_images(frame_S, sobel_kernel_size=7)

    conditions_combination = np.zeros(shape=(frame_S.shape[0], frame_S.shape[1], 3), dtype=np.uint8)

    condition_grad = (direction >= 0.0) & (direction <= 0.78)
    condition_grad &= (mag >= 15) & (mag <= 255)

    condition_color = (frame_S[:, :, 0] >= 96) & (frame_S[:, :, 0] <= 255)

    conditions_combination[:, :, 0] = condition_grad * 255
    conditions_combination[:, :, 1] = condition_color * 255

    binary_mask[condition_color | condition_grad] = 255
    # binary_mask = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), 1)
    binary_mask = cv2.erode(binary_mask, np.ones((2, 2), np.uint8), 1)
    binary_mask = cv2.resize(binary_mask, (frame_clone.shape[1], frame_clone.shape[0]))

    # transformed = transformer.apply(binary_mask, transform_dst_size)
    # transformed = transformer.apply(frame_clone, transform_dst_size)

    # binary_mask = cv2.resize(binary_mask, (frame_clone.shape[1] // 2, frame_clone.shape[0] // 2))

    cv2.imshow('mag', mag)
    cv2.imshow('conditions', conditions_combination)
    cv2.imshow('bin', binary_mask)
    cv2.imshow('S_channel', frame_S)
    cv2.imshow('win', frame)
    #cv2.imshow('transformed', transformed)
    cv2.waitKey(33)

clip.release()
