import os
import csv
import math
import random

import cv2
import numpy as np
import sklearn

folder = 'data_b'

# load samples
samples = []
with open(folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader, None)
    for line in reader:
        # make six sample entries for center left right and their flipped counterpart
        samples.append((-3,line))
        samples.append((-2,line))
        samples.append((-1,line))
        samples.append((0,line))
        samples.append((1,line))
        samples.append((2,line))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32, correction=0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for (i,batch_sample) in batch_samples:
                j = 0
                if (i < 0):
                    j = i + 3
                else:
                    j = i

                name = folder + '/IMG/'+batch_sample[j].split('/')[-1]
                image = cv2.imread(name)
                tmp_cor = 0
                if ( j == 1):
                    tmp_cor = correction
                elif (j == 2):
                    tmp_cor = -correction

                if (i < 0):
                    image = cv2.flip(image, 1)
                    angle = - (float(batch_sample[3]) + tmp_cor)
                else:
                    angle = float(batch_sample[3]) + tmp_cor

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set hyperparameters
batch_size=128
epochs=5
ch_in, row_in, col_in = 3, 160, 320  # in image format
crop_top, crop_bottom, crop_left, crop_right = 50, 20, 0, 0  # in image format
dprob = 0.4

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Imports the Model API
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation, Cropping2D, Lambda, Flatten
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
# Model
data_input = Input((row_in, col_in, ch_in))
cropped_input = Cropping2D(cropping=((crop_top,crop_bottom), (crop_left, crop_right)))(data_input)
resized_input = Lambda(lambda image: tf.image.resize_images(image, (120, 120)))(cropped_input)
#normalized_input = Lambda(lambda x: x/127.5 - 1.)(resized_input)

conv1 = Convolution2D(20, (5, 5), strides=(2,2), padding='valid')(resized_input)
conv1 = BatchNormalization(axis=-1)(conv1)
conv1 = Activation('relu')(conv1)
conv2 = Convolution2D(40, (5, 5), padding='valid')(conv1)
conv2 = BatchNormalization(axis=-1)(conv2)
conv2 = Activation('relu')(conv2)
#maxPooling1 = MaxPooling2D((2,2), strides=(2,2), padding='valid')(conv2)
conv3 = Convolution2D(50, (5, 5), strides=(2,2), padding='valid')(conv2)
conv3 = BatchNormalization(axis=-1)(conv3)
conv3 = Activation('relu')(conv3)
maxPooling2 = MaxPooling2D((5,5), strides=(3,3), padding='valid')(conv3)
conv4 = Convolution2D(60, (3, 3), strides=(2,2), padding='same')(maxPooling2)
conv4 = BatchNormalization(axis=-1)(conv4)
conv4 = Activation('relu')(conv4)
conv5 = Convolution2D(100, (3, 3), padding='valid')(conv4)
conv5 = BatchNormalization(axis=-1)(conv5)
conv5 = Activation('relu')(conv5)

flat = Flatten()(conv5)

fc0 = Dense(120,activation='relu')(flat)
fc0 = Dropout(dprob)(fc0)
fc1 = Dense(60,activation='relu')(fc0)
fc1 = Dropout(dprob)(fc1)
fc2 = Dense(25,activation='tanh')(fc1)
fc2 = Dropout(dprob)(fc2)
fc3 = Dense(10,activation='tanh')(fc2)
estimation = Dense(1)(fc3)

# Creates the model, assuming your final layer is named "predictions"
model = Model(inputs=data_input, outputs=estimation)

model.summary()

model.compile(loss='mse', optimizer='adam')
hist_obj = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=epochs, verbose=1)

model.save('model.h5')
print(hist_obj.history.keys())
print('Loss')
print(hist_obj.history['loss'])
print('Validation Loss')
print(hist_obj.history['val_loss'])



