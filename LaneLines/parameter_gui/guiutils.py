import cv2
import numpy as np


class EdgeFinder:
    def __init__(self, image, filter_size=13, threshold1=45, threshold2=75, verticesX1=450, verticesY1=220, verticesX2=510, verticesY2=220,
                 rho=2, theta=1, threshold=55, min_line_length=100, max_line_gap=50, num_of_pics=6):
        self.image = image
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._verticesX1 = verticesX1
        self._verticesY1 = verticesY1
        self._verticesX2 = verticesX2
        self._verticesY2 = verticesY2
        self._rho = rho
        self._theta = theta * np.pi / 180
        self._threshold = threshold
        self._min_line_length = min_line_length
        self._max_line_gap = max_line_gap
        self._num_of_pics = num_of_pics

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeVerticesY1(pos):
            self._verticesY1 = pos
            self._render()

        def onchangeVerticesX1(pos):
            self._verticesX1 = pos
            self._render()

        def onchangeVerticesY2(pos):
            self._verticesY2 = pos
            self._render()

        def onchangeVerticesX2(pos):
            self._verticesX2 = pos
            self._render()

        def onchangeRho(pos):
            self._rho = pos
            self._render()

        def onchangeTheta(pos):
            self._theta = pos
            self._theta = self._theta * np.pi / 180
            self._render()

        def onchangeThreshold(pos):
            self._threshold = pos
            self._render()

        def onchangeMinLineLength(pos):
            self._min_line_length = pos
            self._render()

        def onchangeMaxLineGap(pos):
            self._max_line_gap = pos
            self._render()

        cv2.namedWindow("Params")
        cv2.resizeWindow("Params", 300, 600)

        cv2.createTrackbar("filter_size", "Params", self._filter_size, 20, onchangeFilterSize)
        cv2.createTrackbar("threshold1", "Params", self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar("threshold2", "Params", self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar("verticesY1", "Params", self._verticesY1, 540, onchangeVerticesY1)
        cv2.createTrackbar("verticesX1", "Params", self._verticesX1, 980, onchangeVerticesX1)
        cv2.createTrackbar("verticesY2", "Params", self._verticesY2, 540, onchangeVerticesY2)
        cv2.createTrackbar("verticesX2", "Params", self._verticesX2, 980, onchangeVerticesX2)
        cv2.createTrackbar("rho", "Params", self._rho, 50, onchangeRho)
        cv2.createTrackbar("theta", "Params", int(self._theta / np.pi * 180), 50, onchangeTheta)
        cv2.createTrackbar("thresholdHough", "Params", self._threshold, 100, onchangeThreshold)
        cv2.createTrackbar("minLineLength", "Params", self._min_line_length, 500, onchangeMinLineLength)
        cv2.createTrackbar("maxLineGap", "Params", self._max_line_gap, 200, onchangeMaxLineGap)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow("Params")
        cv2.destroyWindow("output")

    def filterSize(self):
        return self._filter_size

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def vertices(self):
        return np.array([[(self._verticesX1, self._verticesY1), (self._verticesX2, self._verticesY2)]])

    def rho(self):
        return self._rho

    def theta(self):
        return self._theta * 180 / np.pi

    def thresholdHough(self):
        return self._threshold

    def minLineLength(self):
        return self._min_line_length

    def maxLineGap(self):
        return self._max_line_gap

    def _render(self):
        # convert to grayscale
        self._gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # get image shape and x/y shape
        imshape = self._gray_img.shape
        yshape = 540
        xshape = 960

        # apply smoothing and Canny edge detection
        self._smoothed_img = cv2.GaussianBlur(self._gray_img, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)

        # init mask and output to apply vertices
        mask = np.zeros_like(self._edge_img)
        ignore_mask_color = 255
        self._masked_img = np.zeros((imshape[0], imshape[1]), np.uint8)

        # Apply vertices to each of the concatenated image
        for i in range(0, self._num_of_pics):
            # calculate vertices for i-th image
            vertices = np.array([[((i % 3)*xshape, (int(i % 2)+1)*yshape-10),
                                  ((i % 3)*xshape+self._verticesX1, (int(i % 2))*yshape+self._verticesY1),
                                  ((i % 3)*xshape+self._verticesX2, (int(i % 2))*yshape+self._verticesY2),
                                  (((i % 3)+1)*xshape, (int(i % 2)+1)*yshape-10)]], dtype=np.int32)

            # apply mask and save in masked edges
            cv2.fillPoly(mask, vertices, ignore_mask_color)
            tmp = cv2.bitwise_and(self._edge_img, mask)
            self._masked_img = cv2.bitwise_or(self._masked_img, tmp)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        line_image = np.copy(self.image) * 0  # creating a blank to draw lines on
        lines = cv2.HoughLinesP(self._masked_img, self._rho, self._theta, self._threshold, np.array([]),
                                self._min_line_length, self._max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Create a "color" binary image to combine with line image
        self._color_edges = np.dstack((self._masked_img, self._masked_img, self._masked_img))
        # Draw the lines on the edge image
        self._lines_edges = cv2.addWeighted(self._color_edges, 0.8, line_image, 1, 0)

        # Resize to fit to screen
        imgOut = cv2.resize(self._lines_edges, (1920, 1080))  # Resize image
        cv2.imshow("output", imgOut)  # Show image
