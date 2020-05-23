# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[image1]: ./examples/input.jpg "Input"
[image2]: ./examples/grayscale.jpg "Grayscale"
[image3]: ./examples/smoothed.jpg "Smoothed"
[image4]: ./examples/canny.jpg "After Canny"
[image5]: ./examples/masked.jpg "Masked"
[image61]: ./examples/hough_1.jpg "All Hough lines "
[image62]: ./examples/hough_2.jpg "Selected Hough lines"
[image63]: ./examples/hough_3.jpg "Fitted Lane"
[image7]: ./examples/output.jpg "Output"

---

## The pipeline

At first, all constants are set. These are the filter size for the gaussian blur and the threshold for the canny 
algorithm. The vertices and the parameters for the hough lines. This is done before the actual processing starts.
These parameters are obtained using a extension to [this](https://github.com/maunesh/opencv-gui-helper-tool) GUI proposal from Maunesh Ahir.
The GUI is also included into this project. It uses the pictures in the test_images folder and can be used by
simply running:

`python find_edges.py`

The processing pipeline begins with the calculation of the vertices under consideration of the actual image shape.

The original image
![alt text][image1]
is converted to a grayscale image:
![alt text][image2]

The grayscale image is now smoothed
![alt text][image3]
and the Canny edge detection is applied on the smoothed image:
![alt text][image4]

Now the region of interest is extracted
![alt text][image5]
and the Hough lines are identified (blue).
![alt text][image61]

From all the possible lines, only those with an absolute slope (depending on the left or right marking) between 0.5 and 1.1 are selected.
These boundaries were found by manual analysis. 
![alt text][image62]

The hough lines are separated in left and right data points according to their slope being positive (right) or negative (left). 
A single line is fitted for both sides through all the points. 
The scipy curve fitting function is used to obtain the parameters. 
The resulting lane is depicted in red:
![alt text][image63]

At last, the lines are drawn on the input image:
![alt text][image7]


## Potential shortcomings with your current pipeline
One potential shortcoming would be what would happen when there are tighter curves. The "model" in this case is a straight line and therefore can't deal with curves.
Also the solution is not very robust when thinking about different light or weather conditions such as dust, rain or simply at night.

## Possible improvements to your pipeline
A first improvement would be to use polynomials instead of lines to fit the hough lines. Then the hough line length could be reduced to allow shorter line segments which are then fitted.

Another approach I recently heard of is to use a particle filter with several models for corners, lines, or curves. This is a very robust approach. It can simply replace the hough method without needing to replace other steps.
It is a very complex approach but also very robust.


 
