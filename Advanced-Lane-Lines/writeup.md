## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistortion]: ./output_images/undistortion.png "Undistorted"
[testImage]: ./test_images/test6.jpg "Test Image"
[undistwarped]: ./output_images/warped_straight_lines.png "Warp Example"
[thresholded]: ./output_images/thresholded.jpg "Thresholded"
[fit1]: ./output_images/fit1.jpg "Fit 1"
[fit2]: ./output_images/fit2.jpg "Fit 2"
[out]: ./output_images/output.jpg "Output"
[projectVid]: ./output_videos/project_video.mp4 "Project Video"
[challengeVid]: ./output_videos/challenge_video.mp4 "Challenge Video"
[hardVid]: ./output_videos/harder_challenge_video.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

The code which is used in this task can be found in the folder _solution_. It consits of the _HelperClasses.py_ which includes the main part of the code and the _solution.py_ which does the job of video processing.
There is also the file _parametergui.py_ which is used to estimate the parameter which should be used for the different processes. The GUI accesses the images in the folder _test_images_. For the challenge part, some pictures where added.

### Camera Calibration

The code for this step is contained in the `Calibration` class in the _HelperClasses.py_ file from line 7 to 63.  

First, the parameters are set. They also include the parameters for the perspective transformation. The calibration itself is done in the funciton `calib()`.

I start by loading all files of the _camera_cal_ directory and  preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
The distortion correction can be applied to any image by calling the `getUndist()` function. For the test image I obtained this result: 

![alt text][undistortion]

### Pipeline (single images)

#### Test image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][testImage]

#### Perspective Transform

The source points for the perspective were chosen by hand in a test picture. The transformation is included in the `Calibration` class and can be called using `getWarped()` and `getUnwarped()` for the inverse transformation.

```python
src = np.float32([[srcX[0], srcY[0]], [srcX[1], srcY[1]],
                  [srcX[2], srcY[2]], [srcX[3], srcY[3]]])
dst = np.float32([[dstXoff, srcY[0]+dstYlow], [dstXoff, dstYupp],
                  [imshape[0]-dstXoff, dstYupp], [imshape[0]-dstXoff, srcY[0]+dstYlow]])
```

I chose the following source and destination points where `dstXoff = 300`, `dstYupp = 0`, `dstYlow = 25`:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 258, 683      | 300, 708      | 
| 575, 463      | 300, 0        |
| 711, 463      | 980, 0        |
| 1059, 683     | 980, 708      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][undistwarped]

#### Thresholding and the generation of a binary image

I used a combination of the x gradient, the direction and magnitude of the gradient and the l and s channel of the HLS color space. 
The function to obtain the gradient based binary images is `abs_mag_dir_threshold()` while the function for the hls based binary image is `ls_ch_threshold()`.
The binary s channel threshold is set in dependence of the number thresholded pixels of the l and s binary images. If the number of pixels in the s channel is low and more than 70% of the l channel is covered, then another threshold is selected for the s channel.
This is because of the special concatenation of the thresholded binary images:
```python
combined[(gradx == 1) & (mag_binary == 1) & (dir_binary == 1)] = 1
combined[(combined == 1) | (((gradx == 1) | (mag_binary == 1) | (dir_binary == 1) | (lch_binary == 0)) & (sch_binary == 1))] = 1
combined[(combined == 1) | ((lch_binary == 0) & ((gradx == 1) | (sch_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))))] = 1
```
This allows the extraction of different features under different conditions. Especially under darker conditions the second line extracts additional features which aren't covered by the other statements.

![alt text][thresholded]

#### 

In the `LaneDetection` class there are the functions `fit_poly`, `find_lane_pixels` and `search_around_poly`. If no previous fit was done or the detected lane doesn't make sense or there are not enough points to fit a line, the function `find_lane_pixels` is called.
This function searches line pixels in the binary picture from scratch. It takes the lower quarter of the picture and searches for peaks in the histogram. The peaks are the base for a windowed search in the y direction. 
The height of each window is the same (image_height/9, as there are 9 windows). The width is the margin (set to 100) on both side for the first window on the bottom and is reduced by one fifth until the last window on the top. This is to compensate for the perspective transform which leads to a larger spread pixels in the upper region of the picture.

![alt text][fit1]

When a polynom was fitted previously, the function `search_around_poly` is called. It does principally the same as the `find_lane_pixels` function but doesn't search in windows but in a defined area around the last fit which also decreases to the upper region.
If this method gives results which don't pass the sanity checks for curvature, lane width and own position on the lane. The lane is resetted and a new search with the previous algorithm is conducted in the next step.

![alt text][fit2]


#### Lane Curvature and distance to center of lane

The calculations are done in the function `calculate_curv_dist`. It calculates the radius of both lines and the distance of the vehicle to the center of the lane. It also calculates the lane width.
The function is straightforward and uses code of the previous lectures. The calculation of the distance from lane center
```python
dist = self.xm_per_pix * (leftx + rightx - self._imshape[0])/2
```
where leftx and rightx are the x pixels of the lines at the very bottom of the image.

#### Identification and Drawing

The drawing step is done in the function `clean_result` which draws the green surface and the lines as well as the text on the original image.

![alt text][result]

---

### Pipeline (video)

Here's the [project video](./output_videos/project_video.mp4) and the [challenge video](./output_videos/challenge_video.mp4).

I also tried the [hard challenge](./output_videos/harder_challenge_video.mp4) but didn't succeed on it.

---

### Discussion

The project video itself was working pretty fast. After using the given test images and adapting the parameters using the parametergui to work well on all pictures. The pipeline worked as expected.

The difficulties began when I tried to get the challenge video working. I had to address these major points:

* Different light conditions
* Failure detection
* Smoothing but still react fast enough to changing conditions

I was able to solve the first point by adding some threshold operations like the mentioned adaption of the threshold for dark conditions.
The second step was addressed mainly by detecting the lane width and checking if it is in an accepted range.
I wasn't able to figure out a good approach to solve the third point. Especially for the fast changing conditions in the harder challenge, the weakness of my implementation gets obvious. 
It is only able to smooth if the environment changes slowly enough. An adaptive weighting would be a good idea but wasn't implemented in this solution.

One point in the harder challenge is also the very small radius of some curves. In the used binary image or even in the original image sometimes the lane markings aren't visible at all because they're not in the image.
To solve this a method to compensate for missing lane marks is needed. 
