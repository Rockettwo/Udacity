# **Traffic Sign Recognition** 

## Writeup



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/Histogram.png "Visualization"
[image2]: ./imgs/Grayscale.png "Grayscaling"
[image3]: ./imgs/Training.png "Grayscaling"
[image5]: ./imgs/All5.png "Traffic Signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The histograms are showing the number of samples per class for Training, Validation and Test.
The samples are equally distributed, so for example class 0 (Speed limit - 20 km/h): All three sets contain the same share of samples.

![alt text][image1]

It is important to have the same distribution of classes and corresponding samples in each set and the distribution also should be connected to the reality. If a sample is rare in reality but occurs often in the training data, the model will fit better on that data and a bit worse on the other data.

### Design and Test a Model Architecture

#### 1. Data preprocessing
As a first step, I decided to convert the images to grayscale because it reduces the dimension of each picture, works well and speeds up the training of the network, because it doesnt has to learn this transformation itself.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it is better to train for the neural network if the values have mean zero and small standard deviation. It is also better for transferability.

#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        	| Output        |
|:---------------------:|:------------------------------| :---------:   |
| Input         		| 32x32x1 Grayscale image   	| 32x32x1       |
| Convolution 5x5     	| stride 1, valid padding       | 28x28x12      |
| RELU					|								| 28x28x12      |
| Max pooling 2x2      	| stride 2, valid padding       | 14x14x12      |
| Convolution 5x5     	| stride 1, valid padding       | 10x10x30      |
| RELU					|								| 10x10x30      |
| Convolution 3x3     	| stride 1, same padding        | 10x10x60      |
| RELU					|								| 10x10x60      |
| Max pooling 2x2      	| stride 2, valid padding       | 5x5x60        |
| Flatten             	| convert to single dimension   | 1500          |
| Fully connected		|           					| 400           |
| RELU					|								| 400           |
| Dropout				| rate 0.4						| 400           |
| Fully connected		|           					| 120           |
| RELU					|								| 120           |
| Dropout				| rate 0.4						| 120           |
| Fully connected		|           					| 84            |
| RELU					|								| 84            |
| Dropout				| rate 0.4						| 84            |
| Fully connected		|           					| 43            |
| Softmax				| etc.        					| 43            |
 


#### 3. Training

To train the model, I used an Adam Optimizer which performs better in most cases than SGD or other algorithms. The loss is calculated using cross entropy between softmax result and one hot encoded labels.
For the batch size I choose 120, but as you can see in the picture showing training accuracy (blue) and validation accuracy (red), the accuracy doesn't increase much after 50 epochs.
![alt text][image3]

The learning rate is set to 0.0005 and the batch size 128. To initialize the weights, a gaussian distribution with mean 0 and sigma 0.1 is chosen. The dropout rate was 0.4 so approx. 60% of the connections were cut.

#### 4. My approach

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.5% 
* test set accuracy of 96.7%

I based my network on the LeNet from the lectures. It is a simple and small network, which is capable of image classification tasks. 
I did not have to make many adjustments to increase the performance of the network to beat Facebook. These adjustments were introducing dropout, adding another convolutional layer which themselves were a bit deeper and adding one more fully connected layer.
The training, as well as validation and test set accuracy are over 95% which is tolerable. Most signs will be classified correctly.

### Test a Model on New Images

#### 1. Five German traffic signs
Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image might be difficult to classify because the angle is a bit different from the others. The second image is particularly chosen to be a tough one. I don't expect the network to make a good decision on this one.
All other pictures should be straight forward though.

#### 2. Prediction of new pictures
Here are the results of the prediction:

| Image			        |     Prediction	            | 
|:---------------------:|:-----------------------------:| 
| Priority Road      	| Priority Road   		        | 
| No entry     			| End of speed limit 80 km/h    |
| 60 km/h				| 60 km/h				        |
| General caution	    | General caution               |
| Stop      			| Stop      			        |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is okay because the second picture is an extreme situation and the model has not been trained with any augmentation. To also detect those signs, augmentation is non optional.

#### 3. Certainty 

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For sign 1, 3 and 4 the probability for the correct class was more than 98%, so I won't consider them here. The probability for the priority road sample is 100%. The probability for the general caution is 99.9999%. The probability for the 60 km/h sign is 98.6% followed by a probability of 1.1% for the 80 km/h sign.

For sign 2 the probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .77         			| End of    					| 
| .12     				| No entry 										|
| .04					| End of no passing by vehicles over 3.5 metric tons	|
| .03	      			| End of all speed and passing limits           |
| .02				    | End of no passing      						|
So the second guess is not too bad...

For the fifth image the probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .927         			| Stop   					                    | 
| .037     				| speed limit 80 km/h 							|
| .009					| Turn left ahead	                            |
| .005	      			| No vehicles                                   |
| .003				    | Ahead only      						        |
