import cv2
import os
import numpy as np
import matplotlib.image as mpimg

from guiutils import EdgeFinder


def allPicsConcat():
    concatenated1 = 0
    concatenated2 = 0
    n = 0

    for filename in os.listdir(r"..\\test_images\\"):
        img = mpimg.imread(os.path.join(r"..\\test_images\\", filename))
        # print(np.shape(img))
        if n % 2 == 1:
            if np.size(concatenated1) == 1:
                concatenated1 = img
            else:
                concatenated1 = np.concatenate((concatenated1, img), axis=1)
        else:
            if np.size(concatenated2) == 1:
                concatenated2 = img
            else:
                concatenated2 = np.concatenate((concatenated2, img), axis=1)
        n = n + 1

    concatenated = np.concatenate((concatenated1, concatenated2), axis=0)

    return concatenated


def main():
    img = allPicsConcat()
    num_of_pics = 6

    # img = mpimg.imread(os.path.join(r"..\\test_images\\", "solidWhiteCurve.jpg"))
    # num_of_pics = 1

    edge_finder = EdgeFinder(img, filter_size=13, threshold1=80, threshold2=35,
                             verticesX1=450, verticesY1=320, verticesX2=510, verticesY2=320,
                             rho=1, theta=1, threshold=28, min_line_length=70, max_line_gap=60,
                             num_of_pics=num_of_pics)

    print("Edge parameters:")
    print("GaussianBlur Filter Size: %f" % edge_finder.filterSize())
    print("Threshold1: %f" % edge_finder.threshold1())
    print("Threshold2: %f" % edge_finder.threshold2())
    print("Vertices: ", edge_finder.vertices())
    print("Rho: %f" % edge_finder.rho())
    print("Theta: %f" % edge_finder.theta())
    print("Hough threshold: %f" % edge_finder.thresholdHough())
    print("Min line length: %f" % edge_finder.minLineLength())
    print("Max line gap: %f" % edge_finder.maxLineGap())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
