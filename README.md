# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./output_images/main_image.png "Splash"
[image2]: ./output_images/camera_cal_test.png "Camera Calibration"
[image3]: ./output_images/chessboard_detect.png "Camera Calibration Chessboard"
[image4]: ./output_images/frame_undistort.png "Camera Calibration Video Frame"
[image5]: ./output_images/diff_img.png "Camera Calibration Video Frame Difference Image"
[image6]: ./output_images/warp_perspective.png "Warp Perspective"
[image7]: ./output_images/video.png "YT Link"
[image8]: ./output_images/pipeline_gradients.png "Preprocessing pipeline"
[image9]: ./output_images/pipeline_complete.png "Binarization pipeline"
[image10]: ./output_images/figure_1_col_sum.png "Column sum"
[image11]: ./output_images/figure_2_binarize.png "Rectify signal"
[image12]: ./output_images/figure_3_spikes.png "Identify peaks"
[image13]: ./output_images/sliding_window_positions.png "Identify lane pixels"
[image14]: ./output_images/radius_formula.png "Formula for calculation of radius of curvature"

![Title image][image1]

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


At this point I continued working with all other algorithms on the transformed image, because all necessary information are contained in the warped image. Additionally computation time was reduced, working on the warped image with size (512, 786) instead of the original image size (1280, 720).


## Use color transforms and gradients to create a thresholded binary image

To detect the lane pixels correctly, I created the **LaneDetectorPreprocessor** class (packages/image_preprocessing.py) that is used to create a binary image out of the warped image. In that binary image, only the lane line pixels should be at 255, every other pixel should be 0. 

The pipeline I used is highly inspired by the course content of the project chapter. It uses the following steps:

**Preprocessing pipeline:**

 * Gaussian blur the image with a filter of size (3, 3).
 * Convert image from BGR into HLS colorspace.
 * Apply Contrast Limited Adaptive Histogram Equalization on S-Channel.
 * Use S-Channel for Sobel-Filtering in X and Y.
 * Compute magnitude and direction of gradients images.
 * Use S-Channel for thresholding: t_low < g < t_high.
 * Use magnitude and direction of gradients images for thresholding: 
 *  -> (tm_low < mag < tm_high) AND (td_low < dir < td_high)
     
 * Combine (OR) both thresholded images.
 * Use morphological filter for refinement of structures (erode).
 
I empirically chose the following threshold values:

| Threshold | t_low | t_high | tm_low | tm_high | td_low | td_high |
| --------- | ----- | ------ | ------ | ------- | ------ | ------- |
| Value     | 200   | 255    | 15     | 255     | 0.0    | 0.78    |
 

#### First steps of preprocessing
---
The following image visualizes the first steps of the preprocessing pipeline:
![Preprocessing pipeline][image8]


#### Final steps of preprocessing and binarization 
---
Finally, this image displays the combined thresholding methods that leads to the final binary image:
![Binarization][image9]


Given that final binarized image of the warped version of the orignal frame, the detection of the left and right lane line pixels can start.


## Detect lane pixels and fit to find the lane boundary

For precise detection of the lane lines, I created the class **LaneDetector** (packages/lane_detection.py). This class manages the detection of left and right lane line in the warped and thresholded image. Additionally it holds a list of the last n (n=20) detected lane-polynomials for smoothness filtering. 

### Processing pipeline
The code is inspired by the course material and uses the following functions:

**find_lanes_hist_peaks_initial()**
This function uses the binary mask and computes the column-sums over the lower half of the image. The colum-sum looks like this:

![Colum sum signal][image10]

I use a sliding window with size 30 and compute the sum over this window. If the sum exceeds a threshold of 10000, I set the resulting signal to 1, else 0. With this method I binarize the signal. The result has an offset of filter_size / 2 (orange line is binarized signal, normalized for visualization):

![binarized signal][image11]

Given a binary signal, I use transitions from LOW to HIGH and HIGH to LOW for identifying peaks. Because I know the filter lengths, I can correct the offset of the signal (green signal represents the detected peaks):

![peaks in signal][image12]

The next step is calling the function:
**find_lanes_hist_peaks_filter()**
This function uses the previously computed peaks and uses them to find the two peaks that represent the lane lines. For this purpose, a list of the last m (m=30) peak pairs is used. If this list has enties, the last detected pair is used to find the nearst two peaks of the input list of peaks.

If the list of last detections is empty, a new search is started. I simply take two peaks in the left and the right half of the image, that are nearest to the center. These two peaks are added to the list of last m detections.

#### Sliding window

Next, the function **find_lanes_hist_peaks_lane_positions()** is called for identifying positions of valid lane line pixels. Using the two peaks from the previous function, I shift a sliding window of the size (42, 42) over the image. The scheme of shifting is from bottom to the top of the image.

For every of the two peaks the shifting scheme can be desribed as:
 1. Start at the peak position in the bottom of the image.
 2. Go up vertically in y by 42 pixels (shape of sliding window).
 3. Shift around the previous x position in a range of -shift_range and +shift_range.
 4. A valid position is found, if *intensity_threshold* (0.1) percent of pixels under the window is 1.
 5. Repeat at 2. until reaching the top of the image. 

Given these positions of valid rectangles for each lane, I use these rectangles for masking the left and right lane line pixels in function **find_lanes_hist_peaks_lane_pixels()**. The following image show the result of this pipeline (initial two peaks are inpainted as rectangle as well):

![Visualize lane pixels][image13]

Now that the pixels for the left and right lanes are identified, I use the numpy function **polyfit** for fitting a polynomial of **f(x) = A*x^2 + B*x + C** into these points (in function **fit_poly()**).


#### Sanity Check
To sanity check the detection, I used a thresholding method in **sanity_check_poly()**. Given the left and right polynomials by:

**Polynomial left**: f_left(x) = A_left * x^2 + B_left * x + C_left

**Polynomial right**: f_right(x) = A_right * x^2 + B_right * x + C_right

I calculated abs_diff_B = abs(B_left - B_right) and discarded the polynomials if abs_diff_B exceeds a value of 0.2. If the current detection is dropped, I set the mean of the last n (n=20) polynomials in the filtering method, described in the next part.


#### Filtering 
If the detection of the lane-polynomials succeeds in the current frame, this detected is pushed at the end of the list, and the first elements gets dropped. So I created a floating list of the last n detections and compute the mean over them for smoothness. This happens in the function **filter_poly()**.


## Determine the curvature of the lane and vehicle position with respect to center

To determine the curvature of the current detection, I use the given **xm** and **ym** scaling factors to translate pixel coordinates into meters. I recalculate the polyomials with rescaled pixel coordinates, to get polynomials in world coordinate system. with the help of the given [Tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) and example code, I used the formula

![Radius formula][image14]

to calculate the respective radii in my function **calc_radius_offset_poly()**.

Additionally, having the starting points of the polynomials at the bottom of the image, I computed the distance of the lane center to the image center and converted it into world space, resulting in plausible world coordinates. 

## Generating visualization

For a nice visualization of the lane detection, I draw the polygon of the lane into the warped image, and warp it back to the original image space with the inverse transformation using my class **PerspectiveTransform**. The lane is inpainted in green, if the current frame has a valid (in sense of sanity check) detection, otherwise it will be painted in orange.

I concatenate the 'bird-eye' view image, the S-channel image, the magnitude of gradients image, the composite binarization image and finally an image containing a combination of lane rectangle positions, lane pixels and the polynomials.


## Result

To visualize my result on the project video I uploaded a video on YouTube. The inpainted lane changes the color from green to orange, if the detection in the frame was not confident enough.

The Debug-images in the lower part display (from left to right) the bird eye view of the image part in front of the car (perspective transform), the S-channel from HLS-Colorspace, the magnitude image from edge detection with sobel filtering, a composite binary premasking of lane pixels and the lane detections with corresponding inpainted polynomials.

Please click on the image to watch the video on YouTube:
[![Youtube Link][image7]](https://youtu.be/iSw3WAGySTk "Udacity Self Driving Car ND Project 4 - Advanced Lane Finding")


## Reflection

Some drawbacks and ideas about this project:

 * The described image processing pipeline is quite restricted to the supplied video material.
 * There are a lot of pre-defined thresholds, that make the pipeline less dynamic.
 * Methods of machine learning could help to improve the pipeline.
 * In detail, a classifier could be trained on image patches that contain road lane lines/ road border or not.
 * Furthermore, a semantic segmentation network could be used for detailed scene understanding.
 
