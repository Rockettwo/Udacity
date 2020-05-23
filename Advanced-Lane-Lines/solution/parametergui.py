import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob


class ThreshFinder:
    def __init__(self, images, ksize=3, xThresh=[10,210], yThresh=[10,210], magThresh=[10,70], dirThresh=[0,50],
                 schThresh=[85,255], lchThresh=[0,130], nx=9, ny=6):
        self.images = images
        self.image = cv2.imread(images[0])
        self._undist = self.image
        self._warped = self.image
        self._numOfImgs = len(images)
        self._ksize = ksize  # Choose a larger odd number to smooth gradient measurements
        self._xThresh = xThresh
        self._yThresh = yThresh
        self._magThresh = magThresh
        self._dirThresh = dirThresh
        self._schThresh = schThresh
        self._lchThresh = lchThresh
        self._first = True
        self._nx = nx
        self._ny = ny
        self._srcX = np.array([258, 575, 711, 1059])
        self._srcY = np.array([683, 463, 463, 683])
        self._dstXoff = 250
        self._dstYlow = 25
        self._dstYupp = 0
        self._mtx = []
        self._dist = []
        self._M = []
        self._Minv = []
        self._imshape = self.image.shape[:2][::-1]
        self._margin = 100  # Set the width of the windows +/- margin
        self._minpix = 80  # Set minimum number of pixels found to recente
        self._winDecFactor = 3


        self._leftFit = []
        self._rightFit = []

        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension


        def onchagneImage(pos):
            self.image = cv2.imread(images[pos])
            self._render()

        def onchangeKernelSize(pos):
            if pos % 2 == 0:
                pos = pos + 1
            self._ksize = pos
            self._render()

        def onchangeXThresh(pos, i):
            self._xThresh[i] = pos
            self._render()

        def onchangeYThresh(pos, i):
            self._yThresh[i] = pos
            self._render()

        def onchangeMagThresh(pos, i):
            self._magThresh[i] = pos
            self._render()

        def onchangeDirThresh(pos, i):
            self._dirThresh[i] = pos
            self._render()

        def onchangeSchThresh(pos, i):
            self._schThresh[i] = pos
            self._render()

        def onchangeLchThresh(pos, i):
            self._lchThresh[i] = pos
            self._render()

        def onchangemargin(pos):
            self._margin = pos
            self._render()

        def onchangeminpix(pos):
            self._minpix = pos
            self._render()

        cv2.namedWindow("Params")
        cv2.resizeWindow("Params", 300, 900)

        cv2.createTrackbar("Image no.:", "Params", 0, len(self.images)-1, onchagneImage)
        cv2.createTrackbar("kernel size:", "Params", self._ksize, 20, onchangeKernelSize)
        cv2.createTrackbar("xThresh - Min:", "Params", self._xThresh[0], 255, lambda evt, temp=0: onchangeXThresh(evt, temp))
        cv2.createTrackbar("xThresh - Max:", "Params", self._xThresh[1], 255, lambda evt, temp=1: onchangeXThresh(evt, temp))
        cv2.createTrackbar("yThresh - Min:", "Params", self._yThresh[0], 255, lambda evt, temp=0: onchangeYThresh(evt, temp))
        cv2.createTrackbar("yThresh - Max:", "Params", self._yThresh[1], 255, lambda evt, temp=1: onchangeYThresh(evt, temp))
        cv2.createTrackbar("magThresh - Min:", "Params", self._magThresh[0], 255, lambda evt, temp=0: onchangeMagThresh(evt, temp))
        cv2.createTrackbar("magThresh - Max:", "Params", self._magThresh[1], 255, lambda evt, temp=1: onchangeMagThresh(evt, temp))
        cv2.createTrackbar("dirThresh - Min:", "Params", self._dirThresh[0], 255, lambda evt, temp=0: onchangeDirThresh(evt, temp))
        cv2.createTrackbar("dirThresh - Max:", "Params", self._dirThresh[1], 255, lambda evt, temp=1: onchangeDirThresh(evt, temp))
        cv2.createTrackbar("schThresh - Min:", "Params", self._schThresh[0], 255, lambda evt, temp=0: onchangeSchThresh(evt, temp))
        cv2.createTrackbar("schThresh - Max:", "Params", self._schThresh[1], 255, lambda evt, temp=1: onchangeSchThresh(evt, temp))
        cv2.createTrackbar("lchThresh - Min:", "Params", self._lchThresh[0], 255, lambda evt, temp=0: onchangeLchThresh(evt, temp))
        cv2.createTrackbar("lchThresh - Max:", "Params", self._lchThresh[1], 255, lambda evt, temp=1: onchangeLchThresh(evt, temp))
        cv2.createTrackbar("margin:", "Params", self._margin, 200, onchangemargin)
        cv2.createTrackbar("minPix:", "Params", self._minpix, 100, onchangeminpix)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey()

        cv2.destroyWindow("Params")
        cv2.destroyWindow("output")

    def kernelSize(self):
        return self._ksize

    def xThresh(self):
        return self._xThresh

    def magThresh(self):
        return self._magThresh

    def dirThresh(self):
        return self._dirThresh

    def schThresh(self):
        return self._schThresh

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            return np.zeros_like(img)
        # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * np.absolute(sobel) / np.max(np.absolute(sobel)))
        # 4) Create a binary mask where mag thresholds are met
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary.astype(np.float32)

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate magnitude
        abs_sobxy = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobxy = np.uint8(255 * abs_sobxy / np.max(abs_sobxy))
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(gray)
        mag_binary[(scaled_sobxy > thresh[0]) & (scaled_sobxy <= thresh[1])] = 1

        return mag_binary.astype(np.float32)

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient direction
        # Apply threshold
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate dir
        dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_dir = np.uint8(255 * dir / np.max(dir))
        # 5) Create a binary mask where mag thresholds are met
        dir_binary = np.zeros_like(gray)
        dir_binary[(scaled_dir > thresh[0]) & (scaled_dir <= thresh[1])] = 1

        return dir_binary.astype(np.float32)

    def sl_ch_threshold(self, img, thresh_s=(0, 255), thresh_l=(0, 130)):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_s[0]) & (s_channel <= thresh_s[1])] = 1

        # Threshold color channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1])] = 1

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

        ret, self._mtx, self._dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self._imshape, None, None)

        src = np.float32([[self._srcX[0], self._srcY[0]], [self._srcX[1], self._srcY[1]],
                          [self._srcX[2], self._srcY[2]], [self._srcX[3], self._srcY[3]]])
        xshift = (self._srcX[0] + self._srcX[3] - self._imshape[0]) / 2
        dst = np.float32([[self._dstXoff-xshift, self._srcY[0]+self._dstYlow], [self._dstXoff-xshift, self._dstYupp],
                          [self._imshape[0]-self._dstXoff-xshift, self._dstYupp], [self._imshape[0]-self._dstXoff-xshift, self._srcY[0]+self._dstYlow]])

        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)

    def find_lane_pixels(self, img):
        # HYPERPARAMETERS
        nwindows = 10  # Choose the number of sliding windows

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
            win_xleft_low = leftx_current - int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))  # Update this
            win_xleft_high = leftx_current + int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))  # Update this
            win_xright_low = rightx_current - int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))  # Update this
            win_xright_high = rightx_current + int(self._margin * (1 - window / (self._winDecFactor * (nwindows-1))))  # Update this

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
        self._leftFit = np.polyfit(lefty, leftx, 2)
        self._rightFit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_pts = np.array([[[self._leftFit[0] * ploty[i] ** 2 + self._leftFit[1] * ploty[i] + self._leftFit[2], ploty[i]] for i in range(img.shape[0])]], 'int32')
        right_pts = np.array([[[self._rightFit[0] * ploty[i] ** 2 + self._rightFit[1] * ploty[i] + self._rightFit[2], ploty[i]] for i in range(img.shape[0])]], 'int32')

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        out_img = cv2.polylines(out_img, [left_pts], False, (0, 255, 0))
        out_img = cv2.polylines(out_img, [right_pts], False, (0, 255, 0))

        return out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, left_fit, right_fit, ploty

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
        left_fitx, right_fitx, self._leftFit, self._rightFit, ploty = self.fit_poly(img.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self._margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self._margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self._margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self._margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        left_pts = np.array([[[self._leftFit[0] * ploty[i] ** 2 + self._leftFit[1] * ploty[i] + self._leftFit[2], ploty[i]] for i in range(img.shape[0])]], 'int32')
        right_pts = np.array([[[self._rightFit[0] * ploty[i] ** 2 + self._rightFit[1] * ploty[i] + self._rightFit[2], ploty[i]] for i in range(img.shape[0])]], 'int32')

        # Plots the left and right polynomials on the lane lines
        out_img = cv2.polylines(out_img, [left_pts], False, (0, 255, 0))
        out_img = cv2.polylines(out_img, [right_pts], False, (0, 255, 0))

        return out_img, left_fitx, right_fitx, ploty

    def calculate_curvature(self, y_eval):
        a = self.xm_per_pix/(self.ym_per_pix**2) * self._leftFit[0]
        b = self.xm_per_pix/self.ym_per_pix * self._leftFit[1]
        left_curverad = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / np.absolute(2 * a)

        a = self.xm_per_pix/(self.ym_per_pix**2) * self._rightFit[0]
        b = self.xm_per_pix/self.ym_per_pix * self._rightFit[1]
        right_curverad = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / np.absolute(2 * a)

        return left_curverad, right_curverad

    def clean_result(self, left_fitx, right_fitx, ploty, combined):
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
        final = cv2.warpPerspective(color_warp, self._Minv, (self._imshape[0], self._imshape[1]))
        # Combine the result with the original image
        result = cv2.addWeighted(self._undist, 1, final, 0.3, 0)

        return result

    def _render(self):
        if self._first == True:
            self.calib()
            self._first = False

        imshape = (int(self.image.shape[1] / 2), int(self.image.shape[0] / 2))

        self._undist = cv2.undistort(self.image, self._mtx, self._dist, None, self._mtx)
        self._warped = cv2.warpPerspective(self._undist, self._M, self._imshape, flags=cv2.INTER_LINEAR)

        img = cv2.undistort(self.image, self._mtx, self._dist, None, self._mtx)
        cv2.polylines(img, np.array([[[self._srcX[0], self._srcY[0]], [self._srcX[1], self._srcY[1]],
                             [self._srcX[2], self._srcY[2]], [self._srcX[3], self._srcY[3]]]], 'int32'), False, (0, 255, 0))
        warped = cv2.warpPerspective(img, self._M, self._imshape, flags=cv2.INTER_LINEAR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(result)
        ax2.set_title('Undistorted and Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(self._warped, orient='x', sobel_kernel=self._ksize, thresh=self._xThresh)
        grady = self.abs_sobel_thresh(self._warped, orient='y', sobel_kernel=self._ksize, thresh=self._yThresh)
        mag_binary = self.mag_thresh(self._warped, sobel_kernel=self._ksize, thresh=self._magThresh)
        dir_binary = self.dir_threshold(self._warped, sobel_kernel=self._ksize, thresh=self._dirThresh)
        sch_binary, lch_binary = self.sl_ch_threshold(self._warped, thresh_s=self._schThresh, thresh_l=self._lchThresh)

        combined = np.zeros_like(gradx)
        combined[(gradx == 1) & (mag_binary == 1) & (dir_binary == 1)] = 1
        combined[(combined == 1) | (((gradx == 1) | (mag_binary == 1) | (dir_binary == 1) | (lch_binary == 0)) & (sch_binary == 1))] = 1
        combined[(combined == 1) | ((lch_binary == 0) & ((gradx == 1) | (sch_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))))] = 1

        # Plot the result
        cv2.imshow("input", cv2.resize(self.image, imshape))  # Input
        cv2.imshow("undistorted", cv2.resize(self._undist, imshape))  # Input
        cv2.imshow("warped", cv2.resize(self._warped, imshape))  # warped
        cv2.imshow("gradx", cv2.resize(gradx, imshape))  # gradx
        cv2.imshow("grady", cv2.resize(grady, imshape))  # grady
        cv2.imshow("mag_binary", cv2.resize(mag_binary, imshape))  # mag_binary
        cv2.imshow("dir_binary", cv2.resize(dir_binary, imshape))  # dir_binary
        cv2.imshow("sch_binary", cv2.resize(sch_binary, imshape))  # sch_binary
        cv2.imshow("lch_binary", cv2.resize(lch_binary, imshape))  # lch_binary
        cv2.imshow("output", cv2.resize(combined, imshape))  # Output

        initialLaneFit = self.find_lane_pixels(combined)
        cv2.imshow("fit1", cv2.resize(initialLaneFit, imshape))  # Fit1 Output
        cv2.imwrite("../output_images/fit1.jpg", initialLaneFit)

        secondLaneFit, left_fitx, right_fitx, ploty = self.search_around_poly(combined)
        cv2.imshow("fit2", cv2.resize(secondLaneFit, imshape))  # Fit2 Output
        cv2.imwrite("../output_images/fit2.jpg", secondLaneFit)

        left_curveR, right_curveR = self.calculate_curvature(self.ym_per_pix*(self._imshape[1] - 1))
        print(left_curveR, right_curveR)

        result = self.clean_result(left_fitx, right_fitx, ploty, combined)

        cv2.imshow("result", result)


def main():
    images = glob.glob("../test_images/*.jpg")

    thresholdFinder = ThreshFinder(images)

    print("Edge parameters:")
    print("Kernel size: %f" % thresholdFinder.kernelSize())
    print("x threshold: ", thresholdFinder.xThresh())
    print("mag threshold: ", thresholdFinder.magThresh())
    print("dir threshold: ", thresholdFinder.dirThresh())
    print("sch threshold: ", thresholdFinder.schThresh())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

