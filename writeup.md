# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup/undistortion.png "Undistorted"
[image2_in]: ./writeup/undistortion_in.jpg "Road Input"
[image2_out]: ./writeup/undistortion_out.jpg "Road Transformed"
[image3_masks]: ./writeup/test6.jpg.out.2.thresholding_out.jpg "Binary Example"
[image3_out]: ./writeup/test6.jpg.out.3.perspective_warp_in.jpg "Binary Example"
[image4_in]: ./writeup/straight_lines1.jpg.out.3.perspective_warp_in.jpg "Warp Example"
[image4_out]: ./writeup/straight_lines1.jpg.out.3.perspective_warp_out.jpg "Warp Example"
[image5]: ./writeup/test6.jpg.out.4.lane_search_out.jpg "Fit Visual"
[image6]: ./writeup/test6.jpg.out.jpg "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

You're reading it!

### Camera Calibration

The code for this step is implemented in `calibrate_camera` function in lines 21-57 of the file called `camera.py`.

To calibrate the camera, the `cv2.calibrateCamera()` function is used, which takes object points in the real world coordinates, and corresponding coordinates on the calibration image. The assumption is that the chessboard is flat and does not have depth coordinate, so that `z=0`. The desired object points are generated using `np.mgrid` in the `objp` array. To know the coordinates on the calibration image, the `cv2.findChessboardCorners` is used, which takes the size of the grid to find as a parameter, and returns coordinates in pixels on the image. Then both object points and image points are passed to the `cv2.calibrateCamera()`, which returns camera calibration and distortion coefficients.

The calibration and distortion coefficients are then applied with `cv2.undistort` in lines 87-99 of the `camera.py`.

![alt text][image1]

### Pipeline (single images).

The pipeline is assembled in lines 31-38 of the file `process_video.py` and consists of the following steps:

1. Camera distortion correction
2. Thresholding
3. Perspective warp
4. Lanes search
5. Lanes display

#### 1. Camera distortion correction.

The distortion correction is implemented in the `CameraUndistortion` class in `camera.py`, which uses distortion coefficients loaded with `load_camera_calibration` in line 93 in `process_video.py`.

Here is an example of the input image and same image after undistortion:

![alt text][image2_in]

![alt text][image2_out]

#### 2. Thresholding.

To generate a binary image, I used a combination of color and gradient thresholds. The code is implemented in the class `BinaryThreshold` in `threshold.py` and consists of the following steps:

* Conversion of the image into the HLS (Hue, Lightness, Saturation) color space
* Sobel operator is applied on the Saturation channel to extract saturation gradients, and the following thresholds are combined (`binary_output` line 50 in `threshold.py`) into the binary gradient mask (`binary_sobel` line 57):
  * Gradient values in horizontal and vertical directions (`binary_grad_x` and `binary_grad_x` lines 48-49 in `threshold.py`)
  * Gradient magnitude and angle (`binary_mag` and `binary_dir` lines 46-47 in `threshold.py`)
* Yellow line color is extracted and ANDed with binary gradient mask, it was easier to use Hue value and a bit of Saturation to detect that.
* White line color is extracted and ANDed with binary gradient mask, it was easier to use original RGB values directly.

The results of the yellow and white line masks are ORed to create the resulting binary image.

Here is an example of the output for this step, and example of what each of the mask selected: blue - yellow line mask, green - white line mask, red - Sobel operator mask.

![alt text][image3_out]

![alt text][image3_masks]

#### 3. Perspective transformation.

The code for the perspective transformation is implemented in the `PerspectiveWarp` class in `camera.py`, which accepts source (4 points on the original image) and destination (4 points of the desired rectangle). The transformation relies on the fact that position of the camera is fixed and known. The source points were manually selected on the test image where lane lines were straight, destination points were chosen to be the the size of the image including some offset, so that lane curvative will be also included.

The perspective warp points are defined in the `perspective_warp_config` variable in `process_video.py` file. The same points will be used to warp detected lane lines back to the original (undistorted) image.

Here is an example of the unwarped and warped images, which shows that transformation is working as desired, as the lines appear parallel in the warped image.

![alt text][image4_in]

![alt text][image4_out]

#### 4. Detecting the lane line pixels and fitting with a polynomial.

The lane detection is implemented in the `LaneSearchFitted` in `lanes_fit.py`, which search for each lane separately (lines 324-325 in `lanes_fit.py`).

Each lane line is searched and fitted separately, the code is implemented in the `SingleLaneSearch` class in `lanes_fit.py`. The search is implemented as sliding window search starting in the specific boundaries of the image (the `left` and `right` parameters), and consists of the following steps:

1. Find initial position for the centroids of a line
2. Starting from that position, perform sliding window search using window height and horizontal search margin
3. If nothing was found, keep the values from the previous frame, otherwise fit the image points to polynomial function of power of 2

Searching of the initial position is implemented in the `find_initial_centroids` function in `lanes_fit.py`. It builds histogram of the selected side of the image (left or right), and selects the x position of the highest peak. There are certain drawbacks in such approach, which would be discussed later.

The sliding window search is implemented in the `find_centroids_and_points` function in `lanes_fit.py`. It uses the fixed size windows which are moved up on each iteration (lines 88-101 in `lanes_fit.py`). In each iteration, algorithm selects all the points in the window and adds into the `indices` list. The center of the window is also stored in the `centers` and is returned as part of the result of the function, so that they can be drawn later. The next horizontal position of the window is adjusted if number of points in the window exceeds the threshold. That way window tracks the horizontal slope (curvative) of the line.

The points are then used to fit two polynomial functions: one which represents line in pixels (`SingleLaneSearch.current_lane_func`) and other which represents same line in meters (`SingleLaneSearch.scaled_lane_func`). The latter is needed to simplify calculation of the curvative of the road in meters.

The following picture displays how the results of the search of the algorithm (rectangles representing windows) and fitted lines:

![alt text][image5]

#### 5. Calculating the radius of curvative of the lane and the position of the vehicle with respect to center.

The calculation of the lane parameters is implemented as part of the lane detection in the `LaneSearchFitted` class in `lanes_fit.py`.

The radius of the curve of each lane is calculated in meters using scaled lane function (`r_curve_rad` and `r_curve_rad`, lines 328-329 in `lanes_fit.py`). The `get_curvative` for the power of 2 polynomial line function is implemented in lines 56-57 in `lanes_fit.py`.

The lane center is calculated by calculating cross of the lane curves with bottom of the image and assuming that the camera is mounted in the center of a car, the car shift to the lane center is calculated in `car_shift_m`, and all happens in lines 331-334 in `lanes_fit.py`.

The case when lane lines are turning in different directions is covered in lines 341-352 in `lanes_fit.py`. When it is detected, one of the lines is selected as a good one, other is reconstructed by moving the good line's shape to the position of the bad lane.

The curvative of the lane is calculated as the mean of the curvatives of individual lines in line 362 (`curv_rad`).

#### 6. Plotting back the lane area to the camera image.

Plotting back is implemented in the `DisplayLaneSearchFittedUnwarped` class in `lanes_display.py`. Lane is first drawn as on a separate image (lines 62-67 in `lanes_display.py`) and then warped back using the perspective transformation with source and destination points same as were used during initial warp, but swapped (line 48 in `lanes_display.py`).

Here is an example of the result image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](TODO)

---

### Discussion

The implemented solution is working good enough on the project video, but has the following issues that are not allowing it to pass the challenge videos, and will likely fail in the real road conditions.

#### Issues related to the thresholding

The combination of the Sobel operator and color thresholding cleans up most of the noise from the image, but some of it still remains and fools either initial line position detection or sliding window scanner. It is hard to choose the parameters for the Sobel operator and thresholding, and they seem depend on the road surface condition and overall contrast of the image. For the flat road the parameters I found work quite well, but if there is a dividing wall, or incoming traffic on the opposite side of the lane, or wall on the right side - they all will add noise to the output image, making the job for the next step harder.

The thresholding itself is also not fast enough (or at least its current implementation) - using just color information was about 4 times faster than processing including Sobel filter.

It is also not obvious how it will work during the rain weather conditions.

#### Issues related to the perspective transformation

Perspective transformation introduces its own artifacts, one of which is stretching far objects and shrinking close objects. Because the resolution of the camera, further points would have less details and more noise than close points. From other hand, close points would be less represented in the image because of shrinking, but they had more information initially. As a result, the lane line scanning can be distracted by the noise from far points. The issues with the initial thresholding from the previous step adds up.

To solve that, the following approaches were tried:

1. When searching for the initial positions of the lane lines, the region where the histogram is calculated is initially taken as 1/10th of the height of the image, and if nothing was found there, then gradually increase the height and scan again.
2. When scanning for the line using sliding window, use non linear window height (`find_centroids_and_points_nonlinear`). This allowed to multiply the weight of the points closer to the camera (bottom of the image, which should contain more information, as was discussed before), and less points father (which contain more noise). Another solution for this might be to use weights when performing lane function fitting, but that was not tried.
3. Play with thresholding parameters.
4. Do perspective tranformation first, and then threshold.

The last two approaches did not improve anything.

Approaches 1 and 2 slowed down the processing, but showed some potential. It was decided that further investigation is needed, as they did not improve much on the project video, but for challenge videos, combining them with other ideas can probably help.

#### Issues related to the finding the position of the starting points of the lane lines

The simplest approach where histogram for the lower half of the image is calculated does not work well with the noise coming from the vertical walls or obstacles on the sides of the road, as they are visible as white noise to the left or to the right of the lane and sums up to maximum, producing false peak.

The histogram will also not work well if the lane is very curved, so that lane lines are not going up, but rather bending to the side too much.

Using sliding window convolution for the purpose of the detection (`find_initial_centroids_conv`) did not change things.

The approach that can be tried is to search from the center of the image in the direction of potential position of the lane line (e.g. to the right for the right line), and select the first peak, not the maximum one. The assumption is that there should be no markup in the middle of the lane. The approach will certainly not work when there is marking on the lane (e.g. "yield" sign, or text).

Another issue it is not necessary to perform the search one every frame, but instead the positions from the previous frame can be used and then adjusted if no longer working. This should speed up the processing, and potentially make it more robust.

#### Issues related to fitting of the lane line

This seemed the most straightforward part of the project, but because of the noise introduced on the previous stages, the results were not very satisfactory.

For example, the lane line has thickness, and not all the points from the full width of the lane line are visible after thresholding, so that when quadratic function is fit into the thresholded points, it will have to bend more than needed, especially because there is no much information in the upper part of the image (due to perspective), and any noise there can change the resulting curvative of the line dramatically.

When more than one road turn is visible in the camera, fitting quadratic function is no longer satisfactory. I tried to use splines, but found that they required more initial points to fit, then it is easier to work with the quadratic functions to determine the curvative radius, and to smooth the errors. But it seems as the right direction.

#### Issues related to finding outliers and smoothing

Sometimes it seems just impossible to detect the lane lines properly, it can be that it is better to just drop the current frame and take the next one. The simplistic approach where current lane line is simply averaged according to the last several frames did not work very well on turns, and the lane lines were displayed with delay. Potentially better approach would be to filter out outliers first and keep previous lane parameters until some threshold, and update after threshold is exceeded, meaning that new lane parameters are now valid.


