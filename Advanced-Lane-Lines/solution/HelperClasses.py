import cv2
import numpy as np
import glob as glob
import matplotlib.image as mpimg


class Calibration:
    def __init__(self,nx=9, ny=6):
        self._nx = nx
        self._ny = ny
        self._srcX = np.array([258, 575, 711, 1059])
        self._srcY = np.array([683, 463, 463, 683])
        self._dstXoff = 300
        self._dstYlow = 25
        self._dstYupp = 0
        self._mtx = []
        self._dist = []
        self._M = []
        self._Minv = []
        self._imshape = (1, 1)

        self.calib()

    def calib(self):
        # read all calib images and set chessboard numbers
        images = glob.glob('../camera_cal/calib*.jpg')
        nx = 9
        ny = 6

        # prepare object/image points
        objpoints = []
        imgpoints = []

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x,y coordinates

        for fname in images:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

            self._imshape = img.shape[:2][::-1]

        ret, self._mtx, self._dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self._imshape, None, None)

        src = np.float32([[self._srcX[0], self._srcY[0]], [self._srcX[1], self._srcY[1]],
                          [self._srcX[2], self._srcY[2]], [self._srcX[3], self._srcY[3]]])
        xshift = (self._srcX[0] + self._srcX[3] - self._imshape[0]) / 2
        dst = np.float32([[self._dstXoff-xshift, self._srcY[0]+self._dstYlow], [self._dstXoff-xshift, self._dstYupp],
                          [self._imshape[0]-self._dstXoff-xshift, self._dstYupp], [self._imshape[0]-self._dstXoff-xshift, self._srcY[0]+self._dstYlow]])

        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)

    def getUndist(self, img):
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)

    def getWarped(self, img):
        return cv2.warpPerspective(img, self._M, self._imshape, flags=cv2.INTER_LINEAR)

    def getUnwarped(self, img):
        return cv2.warpPerspective(img, self._Minv, self._imshape)


class LaneDetection:
    def __init__(self, ksize=3, xThresh=[10,210], magThresh=[10,70], dirThresh=[0,50], schThresh=[85,255], lchThresh=[0,130],):
        self._warped = []
        self._imshape = (1, 1)
        self._ksize = ksize  # Choose a larger odd number to smooth gradient measurements
        self._xThresh = xThresh
        self._magThresh = magThresh
        self._dirThresh = dirThresh
        self._schThresh = schThresh
        self._lchThresh = lchThresh

        self._margin = 120  # Set the width of the windows +/- margin
        self._minpix = 70  # Set minimum number of pixels found to recenter window

        self._resetted = True
        self._frameCounter = 0

        self._numOfWins = 9
        self._winDecFactor = 1000
        self._lineAvgOver = 4

        self._leftFit = np.array([])
        self._leftBestFit = np.array([])
        self._rightFit = np.array([])
        self._rightBestFit = np.array([])

        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700  # meters per pixel in x dimension

        self._calibration = Calibration()

    def abs_mag_dir_threshold(self, sobel_kernel=3, thresh_x=(0, 255), thresh_mag=(0, 255), thresh_dir=(0, 255)):
        # 1) Convert to grayscale
        gray = cv2.cvtColor(self._warped, cv2.COLOR_RGB2GRAY)

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # xsobel solely
        scaled_sobelx = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
        # 4) Create a binary mask where mag thresholds are met
        grad_binary = np.zeros_like(scaled_sobelx)
        grad_binary[(scaled_sobelx > thresh_x[0]) & (scaled_sobelx <= thresh_x[1])] = 1

        # mag solely
        abs_sobxy = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobxy = np.uint8(255 * abs_sobxy / np.max(abs_sobxy))
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(gray)
        mag_binary[(scaled_sobxy > thresh_mag[0]) & (scaled_sobxy <= thresh_mag[1])] = 1

        # dir solely
        dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_dir = np.uint8(255 * dir / np.max(dir))
        # 5) Create a binary mask where mag thresholds are met
        dir_binary = np.zeros_like(gray)
        dir_binary[(scaled_dir > thresh_dir[0]) & (scaled_dir <= thresh_dir[1])] = 1

        return grad_binary.astype(np.float32), mag_binary.astype(np.float32), dir_binary.astype(np.float32)

    def ls_ch_threshold(self, thresh_s=(0, 255) , thresh_l=(0, 255)):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(self._warped, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Threshold l channel
        l_binary = np.zeros_like(s_channel)
        l_binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1])] = 1

        # Threshold s channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1

        # adjust threshold if not enough pixels
        if (np.count_nonzero(s_binary == 1) < self._imshape[0] * self._imshape[1] * 0.025) \
                & (np.count_nonzero(l_channel < 100) > self._imshape[0] * self._imshape[1] * 0.7):
            s_binary = np.zeros_like(s_channel)
            s_binary[(s_channel >= 20) & (s_channel <= 50)] = 1
            l_binary = np.zeros_like(l_channel)
            l_binary[(l_channel <= 140)] = 1
        elif np.count_nonzero(s_binary == 1) < self._imshape[0] * self._imshape[1] * 0.025:
            s_binary = np.zeros_like(s_channel)
            s_binary[(s_channel >= 25) & (s_channel <= 155)] = 1

        return s_binary.astype(np.float32), l_binary.astype(np.float32)

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        if self._resetted == True:
            self._leftFit = np.polyfit(lefty, leftx, 2)
        elif leftx.size & lefty.size & self._leftFit.size:
            self._leftFit = ((self._lineAvgOver - 1) * self._leftFit + np.polyfit(lefty, leftx, 2)) / self._lineAvgOver
        elif leftx.size & lefty.size:
            self._leftFit = np.polyfit(lefty, leftx, 2)
        elif self._leftFit.size:
            self._resetted = True
        else:
            self._leftFit = np.array([0, 0, 0])

        if self._resetted == True:
            self._rightFit = np.polyfit(righty, rightx, 2)
        elif rightx.size & righty.size & self._rightFit.size:
            self._rightFit = ((self._lineAvgOver - 1) * self._rightFit + np.polyfit(righty, rightx, 2)) / self._lineAvgOver
        elif rightx.size & righty.size:
            self._rightFit = np.polyfit(righty, rightx, 2)
        elif self._rightFit.size:
            self._resetted = True
        else:
            self._rightFit = np.array([0, 0, 0])

        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        left_fitx = self._leftFit[0] * ploty ** 2 + self._leftFit[1] * ploty + self._leftFit[2]
        right_fitx = self._rightFit[0] * ploty ** 2 + self._rightFit[1] * ploty + self._rightFit[2]

        return left_fitx, right_fitx, ploty

    def find_lane_pixels(self, img):
        # Take a histogram of the bottom third of the image
        histogram = np.sum(img[img.shape[0] // 3:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        xoffset = 50
        leftx_base = np.argmax(histogram[:midpoint-xoffset])
        rightx_base = np.argmax(histogram[midpoint+xoffset:]) + midpoint+xoffset

        # HYPERPARAMETERS
        nwindows = 9  # Choose the number of sliding windows

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))
            win_xleft_high = leftx_current + int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))
            win_xright_low = rightx_current - int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))
            win_xright_high = rightx_current + int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # recenter next window
            if len(good_left_inds) > self._minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self._minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial
        self.fit_poly(img.shape, leftx, lefty, rightx, righty)

    def search_around_poly(self, img):
        # Grab activated pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin_poly = self._margin * (1 - (1 - nonzeroy / self._imshape[1]) / self._winDecFactor)

        # Set the area of search based on activated x-values
        left_lane_inds = ((nonzerox > (self._leftFit[0] * (nonzeroy ** 2) + self._leftFit[1] * nonzeroy + self._leftFit[2] - margin_poly)) &
                          (nonzerox < (self._leftFit[0] * (nonzeroy ** 2) + self._leftFit[1] * nonzeroy + self._leftFit[2] + margin_poly)))

        right_lane_inds = ((nonzerox > (self._rightFit[0] * (nonzeroy ** 2) + self._rightFit[1] * nonzeroy + self._rightFit[2] - margin_poly)) &
                           (nonzerox < (self._rightFit[0] * (nonzeroy ** 2) + self._rightFit[1] * nonzeroy + self._rightFit[2] + margin_poly)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(img.shape, leftx, lefty, rightx, righty)

        return left_fitx, right_fitx, ploty

    def calculate_curv_dist(self, y_eval, leftx, rightx):
        a = self.xm_per_pix/(self.ym_per_pix**2) * self._leftFit[0]
        b = self.xm_per_pix/self.ym_per_pix * self._leftFit[1]
        left_curverad = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / np.absolute(2 * a)

        a = self.xm_per_pix/(self.ym_per_pix**2) * self._rightFit[0]
        b = self.xm_per_pix/self.ym_per_pix * self._rightFit[1]
        right_curverad = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / np.absolute(2 * a)

        dist = self.xm_per_pix * (leftx + rightx - self._imshape[0])/2

        lane_width = self.xm_per_pix * (rightx - leftx)

        return left_curverad, right_curverad, dist, lane_width

    def clean_result(self, left_fitx, right_fitx, ploty, img, combined, curverad, dist, lane_width):
        # Create an image to draw the lines on
        combined_zero = np.zeros_like(combined).astype(np.uint8)
        color_warp = np.dstack((combined_zero, combined_zero, combined_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        final = self._calibration.getUnwarped(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, final, 0.3, 0)

        text = f"radius: {curverad:8.2f}m \ndistance: {dist:1.3f}m  \nwidth: {lane_width:1.3f}m "

        y0, dy = 50, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(result, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return result

    def processImage(self, img):
        self._imshape = img.shape[:2][::-1]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self._warped = self._calibration.getWarped(self._calibration.getUndist(img))

        # Apply each of the thresholding functions
        gradx, mag_binary, dir_binary = self.abs_mag_dir_threshold(sobel_kernel=self._ksize,
                                                                   thresh_x=self._xThresh, thresh_mag=self._magThresh,
                                                                   thresh_dir=self._dirThresh)
        sch_binary, lch_binary = self.ls_ch_threshold(thresh_s=self._schThresh, thresh_l=self._lchThresh)

        # combine thresholding functions
        combined = np.zeros_like(gradx)
        combined[(gradx == 1) & (mag_binary == 1) & (dir_binary == 1)] = 1
        combined[(combined == 1) | (((gradx == 1) | (mag_binary == 1) | (dir_binary == 1) | (lch_binary == 0)) & (sch_binary == 1))] = 1
        combined[(combined == 1) | ((lch_binary == 0) & ((gradx == 1) | (sch_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))))] = 1

        if self._resetted:
            self.find_lane_pixels(combined)
            self._resetted = False

        left_fitx, right_fitx, ploty = self.search_around_poly(combined)

        left_curveR, right_curveR, dist, lane_width = self.calculate_curv_dist(self.ym_per_pix*(self._imshape[1] - 1), left_fitx[-1], right_fitx[-1])
        print(left_curveR, right_curveR, dist, lane_width)

        if (left_curveR < 50) | (right_curveR < 50) | (dist > 1.5) | (dist < -1.5) | (lane_width < 2) | (lane_width > 4):
            self._resetted = True

        result = self.clean_result(left_fitx, right_fitx, ploty, img, combined, (left_curveR + right_curveR)/2, dist, lane_width)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
