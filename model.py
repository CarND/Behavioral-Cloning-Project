import os
import csv
from scipy import ndimage
import cv2 
import math 

import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, Cropping2D, MaxPooling2D
# from keras.layers.convolutional import Convolution2D#, MaxPooling2D
import numpy as np 
from sklearn.utils import shuffle

# import tensorflow.compat.v1 as tf

from keras.callbacks import ModelCheckpoint
from keras import backend as K


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # create adjusted steering measurements for the side camera images - tune experimentally
                correction = 0.1 

                left_name = './data/data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = center_angle + correction

                right_name = './data/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = center_angle - correction

                # mirror turns to combat potential bias 
                mirror_center_image = np.fliplr(center_image)
                mirror_center_angle = center_angle * -1
                mirror_left_image = np.fliplr(left_image)
                mirror_left_angle = left_angle * -1
                mirror_right_image = np.fliplr(right_image)
                mirror_right_angle = right_angle * -1

                # add left and right image and flip all 3 images
                images.extend((left_image, 
                               right_image, 
                               mirror_center_image, 
                               mirror_left_image, 
                               mirror_right_image))
                
                # add associated angles
                angles.extend((left_angle, 
                               right_angle, 
                               mirror_center_angle, 
                               mirror_left_angle, 
                               mirror_right_angle))

            X_train = np.array(images)
            y_train = np.array(angles)
            # print("X, y train type ", type(X_train), type(y_train))
            yield (X_train, y_train)

if __name__ == "__main__":
    lines = []

    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # Set our batch size
    batch_size=32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    # original image
    ch, row, col = 3, 160, 320 

    model = Sequential()
    # crop input image (top, bottom) 3@160x320, output 3@65x320
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(row, col, ch))) 
    model.add(Lambda(lambda x: (2*x / 255) - 1.0))
    model.add(Conv2D(24, (5,5), activation="relu", strides=(2,2)))
    model.add(Conv2D(36, (5,5), activation="relu", strides=(2,2)))
    model.add(Conv2D(48, (5,5), activation="relu", strides=(2,2)))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adamax')

    model.summary()

    model.fit_generator(train_generator, 
                steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                validation_data=validation_generator, 
                validation_steps=math.ceil(len(validation_samples)/batch_size), 
                epochs=3, verbose=1)








