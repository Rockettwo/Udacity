# **Behavioral Cloning** 

##### My submission

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[rec_1]: ./examples/curve_right.jpg "Recovery Image"
[rec_2]: ./examples/curve_right_end.jpg "Recovery Image"
[normal]: ./examples/center.jpg "Normal Image"
[left]: ./examples/left.jpg "Left Image"
[right]: ./examples/right.jpg "Right Image"
[cropped]: ./examples/cropped.png "Cropped Image"
[flipped]: ./examples/flipped.png "Flipped Image"
[resized]: ./examples/resizedd.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The input data is first cropped and then rescaled using a Keras lambda function (_model.py_ lines 84-87).
The first part of my model consists of a convolution neural network with 5x5 as well as 3x3 filter sizes and depths between 20 and 100 (_model.py_ lines 89-105). After each convolution a batch normalization is done and ReLU and tanh activation applied to introduce nonlinearity.

The second part is a fully connected layer consisting of Keras dense layers with one ReLU and one tanh activation. There are also dropout layers included which drop connections with a probability of 40%.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 110&112&114). 

The model was trained and validated on different data sets which are all collected in one folder to ensure that the model was not overfitting (code line 14f.). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 121). The batch size was chosen to 128 and the model is trained in 5 epochs. 

#### 4. Appropriate training data

Training data was at first manually collected and is not included in the submission. At data acquisition the vehicle was kept in the center of the lane. I used data from both environments and drove both in starting direction and backwards.
After some training steps I sadly lost the data because of an unexpected connection issue, so the final model was trained on the given data set with some more recoveries from left and right.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to avoid overfitting by collecting enough different data and keep the net structure small and simple.

My first step was to use a convolution neural network model similar to the proposed NVIDIA net. But I changed many of the given parameters, added two pooling layers and added the batch normalization. In the fully connected part the proposed ReLU activation was not used for the last layers as otherwise there would not be any negative output. Instead the tanh activation was implemented for the last layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% - 20%). I found that my first approach had no problem with overfitting but didn't generalize very well. This was because I forgot the initial normalization which is now included in the Keras lambda function.

I noticed that using both input and batch normalization is not working well together. I didn't understand why it makes that big of a difference but I decided to skip input normalization.

The pooling layer is mainly to reduce the number of parameters and also reduce the complexity of the net. I decided to use MaxPooling instead of Average pooling because it worked slightly better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially after the bridge when the right lane markings are missing.
To improve the driving behavior in these cases, I simply recorded just this scene three more times. I also added some more recoveries from the left and the right side on the normal track and in the curves with red-white lane markings.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-113) consisted of a convolution neural network with the following layers and layer sizes:

| layer             | output shape  | kernel    | stride    | info      |
|-------------------|---------------|-----------|-----------|-----------|
|  input_1          |  160,320,3    |           |           |           |
|  cropping2d_1     |  90,320,3     |           |           | (50,20),(0,0)|
|  lambda_1         |  120,120,3    |           |           |           |
|  conv2d_1         |  58,58,20     | 5x5x20    | 2         | valid     |
|  batch_norm_1     |  58,58,20     |           |           | axis = -1 |
|  activation_1     |  58,58,20     |           |           | ReLU      |
|  conv2d_2         |  54,54,40     | 5x5x40    | 1         | valid     |
|  batch_norm_2     |  54,54,40     |           |           | axis = -1 |
|  activation_2     |  54,54,40     |           |           | ReLU      |
|  conv2d_3         |  25,25,50     | 5x5x50    | 2         | valid     |
|  batch_norm_3     |  25,25,50     |           |           | axis = -1 |
|  activation_3     |  25,25,50     |           |           | ReLU      |
|  max_pooling2d_2  |  7,7,50       | 5x5x1     | 3         | valid		|
|  conv2d_4         |  4,4,60       | 3x3x60    | 2         | same      |
|  batch_norm_4     |  4,4,60       |           |           | axis = -1 |
|  activation_4     |  4,4,60       |           |           | ReLU      |
|  conv2d_5         |  2,2,100      | 3x3x100   | 1         | valid     |
|  batch_norm_5     |  2,2,100      |           |           | axis = -1 |
|  activation_5     |  2,2,100      |           |           | ReLU      |
|  flatten_1        |  400          |           |           |           |
|  dense_1          |  120          |           |           | ReLU      |
|  dropout_1        |  60           |           |           | prob = 0.4|
|  dense_2          |  60           |           |           | ReLU      |
|  dropout_2        |  60           |           |           | prob = 0.4|
|  dense_3          |  25           |           |           | tanh      |
|  dropout_3        |  25           |           |           | prob = 0.4|
|  dense_4          |  10           |           |           | tanh		|
|  dense_5          |  1            |           |           |           |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][normal]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and also is able to keep the track in curves. These images show what a recovery looks like starting from the right side and moving to center:

![alt text][rec_1]
![alt text][rec_2]

Then I also collected some data from track 2. But I just collected data from the beginning to keep the total data small.

To augment the data sat, I also flipped images and angles thinking that this would generalize better, because the data would have a leftish drift otherwise. For example, here is the flipped image of the center line driving above:

![alt text][flipped]

I also took left and right pictures into account to generalize better and extend the total number of datasets. I therefor added (subtracted) a factor of 0.2 if the picture was taken left (right) of the center.
The left and right pictures are:

![alt text][left]
![alt text][right]

After the collection process, I had 91680 data points. I didn't have to preprocessed this data because it is cropped, rescaled within the convolutional network.
After cropping the image looks:

![alt text][cropped]

And after rescaling to 120x120:

![alt text][resized]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as it already worked well enough and generalied pretty good. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Annotation

Training data can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

FFMPEG is not included in the workspace but is required by video.py. In model_bak.h5 my first network approach with the lost training data is saved. It also works pretty well and is visualized in video_bak.mp4. 



