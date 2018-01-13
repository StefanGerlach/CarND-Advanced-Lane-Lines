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

from moviepy.editor import ImageSequenceClip

from packages.camera_calibration import CameraCalibration
from packages.image_transform import PerspectiveTransform
from packages.image_preprocessing import LaneDetectorPreprocessor
from packages.lane_detection import LaneDetector


""" Definition of some globals """

# For intermediate outputs
output_path = 'output_images'
output_video_frames = 'video_frames'

# For Video output
output_video_file = 'output_video.mp4'

# For input of camera calibration
cal_path = 'camera_cal'
camera_id = 'udacity_video_camera'

# Directory for image transformations
transform_path = 'image_transform'

# Eye-balled position for perspective transformation
transform_dst_size = (512, 768)
transform_pts_a = [(565, 478), (740, 478), (1095, 675), (315, 675)]
transform_pts_b = [(128, 384), (384, 384), (384, 768), (128, 768)]

# Video files
video_file = 'project_video.mp4'
# video_file = 'challenge_video.mp4'
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

# Instantiate Preprocessor
preprocessor = LaneDetectorPreprocessor()

# Instantiate LaneDetector
detect = LaneDetector()

# Open video file
clip = cv2.VideoCapture(video_file)


# Iterate all frames of the video
frame_id = 0
cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

while clip.isOpened():
    _, frame = clip.read()
    if frame is None:
        break

    frame_id += 1

    # Undistort the camera radial distortion
    frame = camera_cal.undistort_image(frame, calibration_dict=cal)

    # Transform the camera image into bird-eye perspective and crop
    frame_bird_eye = transformer.apply(frame, transform_dst_size)

    # Preprocess the bird-eye image into a binary image with lanes as ones
    binary_mask, s_channel, magnitude_img, conditions_combination = preprocessor.preprocess(frame_bird_eye)

    # Compute the lane polynomials, the curvature and the offset from binary_mask
    binary_mask, lane_image, curvature, offset = detect.find_lanes_hist_peaks(binary_mask)

    # Warp the lane image back to original perspective and add weighted
    lane_image = transformer.apply_inv(lane_image, (frame.shape[1], frame.shape[0]))
    lane_image = cv2.addWeighted(frame, 1, lane_image, 0.3, 0)

    # Paint the text to the image
    offset_dir_text = ' left from center' if offset > 0 else ' right from center'
    cv2.putText(lane_image, 'Radius of Curvature = ' + str(int(curvature)) + '(m)', (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 2)
    cv2.putText(lane_image, 'Vehicle is ' + str(abs(int(offset * 100) / 100.0)) + 'm' + offset_dir_text, (25, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 2)

    # Create a complete image of the intermediate debug-images
    inpaint = np.zeros(shape=(1656, 1920, 3), dtype=np.uint8)
    inpaint[0:1080, 0:1920, :] = cv2.resize(lane_image, (1920, 1080))

    sub_imgs = [frame_bird_eye, s_channel, magnitude_img, conditions_combination, binary_mask]
    sub_imgs_desc = ['Bird eye view', 'Saturation Channel', 'Edges Magnitude', 'Lane Pixels', 'Lane Detection']
    sub_img_size = (576, 384)

    for n in range(len(sub_imgs)):
        # Draw text of description in the image
        cv2.putText(sub_imgs[n], sub_imgs_desc[n], (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        # Put the sub image in final image
        y_start = int(n * sub_img_size[1])
        inpaint[1080:1656, y_start:y_start+sub_img_size[1], :] = cv2.resize(sub_imgs[n], (sub_img_size[1], sub_img_size[0]))

    # Put Border for 16:9 scaling
    inpaint = cv2.resize(cv2.copyMakeBorder(inpaint, 0, 0, 512, 512, cv2.BORDER_CONSTANT), (1920, 1080))

    cv2.imshow('Lane Detection', inpaint)
    cv2.waitKey(1)

    # Save this Frame
    if os.path.exists(output_video_frames) is False:
        os.makedirs(output_video_frames)

    cv2.imwrite(os.path.join(output_video_frames, 'frame_' + str(frame_id).zfill(4)+'.png'), inpaint)

clip.release()

# Create output video
print('Creating Video.')

video_file = output_video_file
clip = ImageSequenceClip(output_video_frames, fps=30)
clip.write_videofile(video_file)
