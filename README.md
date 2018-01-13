# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image2]: ./output_images/camera_cal_test.png "Camera Calibration"
[image3]: ./output_images/chessboard_detect.png "Camera Calibration Chessboard"
[image4]: ./output_images/frame_undistort.png "Camera Calibration Video Frame"
[image5]: ./output_images/diff_img.png "Camera Calibration Video Frame Difference Image"

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



