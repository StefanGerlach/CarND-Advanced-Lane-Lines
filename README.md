# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image2]: ./output_images/camera_cal_test.png "Camera Calibration"
[image3]: ./output_images/chessboard_detect.png "Camera Calibration Chessboard"
[image4]: ./output_images/frame_undistort.png "Camera Calibration Video Frame"
[image5]: ./output_images/diff_img.png "Camera Calibration Video Frame Difference Image"
[image6]: ./output_images/warp_perspective.png "Warp Perspective"

In this repository I describe my approach to write a software pipeline that identifies the lane of the road in front of a car in a video file. The precise requirements of this project are:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In the following writeup I describe how I managed to solve the requirements.


##  Code description

This overview describes the project structure and modules:

* packages/camera_calibration.py containing a class for camera calibration 
* packages/image_color_transform.py containing a class that serves as small wrapper for OpenCV cvtColor function
* packages/image_transform.py class PerspectiveTransform for perspective transformations (create, load, save)
* packages/image_gradient.py.py class Gradient for computing edge-images and meta images like magnitude and direction of gradients
* packages/image_preprocessing.py.py class LaneDetectorPreprocessor for creating the binary mask of lane pixels
* packages/lane_detection.py class LaneDetector for detection and filtering of the lane polynomials
* main.py the main pipeline putting it all together


##  Compute the camera calibration matrix

To correctly identifiy objects and compute their properties with respect to real world coordinates, it is necessary to note, that most cameras show a distortion in their optical projections. These distortions are often seen as barrel or pincushion (both are radial)  distortions and modify the shape of objects, so that precise detections are made difficult.

The solution is to compute a camera calibration matrix and the distortion coefficients to undistort the images of the video frame by frame. To do this, there is a helpfull pipeline described in the [Udacity Camera Calibration Repositoy that I forked](https://github.com/StefanGerlach/CarND-Camera-Calibration]). 

For computing the camera matrix and distortion coefficients, [OpenCV](https://opencv.org/) offers some very easy to use functions. The rough pipeline for this is to: 
  * Print out a chessboard pattern and to fixate it on a plane. For this pattern, the real world coordinates are known, so the relation between the image points (camera projection) and real world points can be done.
  * Make some images of the chessboard from different angles, distances and in different imageparts with the camera, that should be calibrated.
  * Load every image and detect the chessboard pattern with the OpenCV function *findChessboardCorners()*. If the chessboard was correctly detected, one can visualize them with *drawChessboardCorners()*:  
![Camera Calibration Chessboard Detection][image3]

  * Collect all point correspondences of a synthetic grid of 3D points (with Z = 0 because the chessboard is a plane) that represent the chessboard corners in the real world, and the detected chessboard corners in every image.
  * With the help of the OpenCV function *calibrateCamera()* the necessary matrix and distortion coefficients are computed.
  * Finally the function *undistort* uses them to undistort an image. The following image visualizes the impact of radial camera distortion and the output of the undistorted image:
  ![Camera Calibration][image2]


## Apply a distortion correction to raw images

Using the computed camera calibration matrix and distortion coefficients, the image processing pipeline of this projects start with undistorting every image, before any other image process routines begin:

![Camera Calibration Frame by Frame][image4]

To better visualize the difference between these two images, the following shows the absolute difference image of original and undistorted frame:

![Camera Calibration Difference Image][image5]

To implement this step, I created the **CameraCalibration** class in packages\camera_calibration.py. It uses the described OpenCV functions to compute and apply the camera matrix and distortion coefficients.


## Apply a perspective transform for "birds-eye view"

To better detect lane lines and reduce the area of the image to look for lane lines, it is usefull to create a warped version of the original frame. This warped version looks like from "the eyes of a bird", it displays the image content from the "top". This warping is called perspective transformation and can be done by multiplying the image pixel positions by a perspective transformation matrix.

The class **PerspectiveTransform** in this repositiory (/packages/image_transform.py) uses OpenCV functions to compute a perspective transformation matrix (and its inverse). All that is needed to do this are 4 points in the source image and 4 points (e.g. defining a rectangle) in the destination image. With these 4 point correspondences and the OpenCV function *getPerspectiveTransform()* the transformation matrix is computed.

I used 4 points in the original image, that define a rectangle that lays on the road lane and 4 points in the destination image, that define a rectangle in the image plane. The next image visualizes these rectangles with green lines and the point correspondences with red lines:

![Perspective Transformation][image6]


At this point I continued working with all other algorithms on the transformed image, because only the necessary image information are contained in the warped image. Additionally computation time was reduced, working on the warped image with size (512, 786) instead of the original image size (1280, 720).


