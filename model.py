import pandas as pd
import cv2
import numpy as np
import os
import re

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MAX_STEERING = 1
MIN_STEERING = -1

BATCH_SIZE = 64

steering_correction = 0.1

# Get data set 1 (both track both directions, center line driving)
driving_log_01 = pd.read_csv('./data/01_t1_t2_forw_backw.csv',
                             names=['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'brake', 'speed']
                             )[:100]
print(driving_log_01.shape)

# Get data set 2 (special driving maneuvers)
driving_log_02 = pd.read_csv('./data/02_special_driving_maneuvers.csv',
                             names=['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'brake', 'speed']
                             )[:100]
print(driving_log_02.shape)

# Get data set 3 (udacity dataset)
# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
# Data set includes header
driving_log_03 = pd.read_csv('./data/03_udacity_data.csv',
                             names=['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'brake', 'speed'],
                             skiprows=[0]
                             )[:100]
print(driving_log_03.shape)

# Get data set 4 (joystick center line driving track 1)
driving_log_04 = pd.read_csv('./data/04_joystick_track1.csv',
                             names=['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'brake', 'speed']
                             )[:100]
print(driving_log_04.shape)

# Get data set 4 (joystick center line driving track 2)
driving_log_05 = pd.read_csv('./data/05_joystick_track2.csv',
                             names=['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'brake', 'speed']
                             )[:100]
print(driving_log_05.shape)

# Pool data
frames = [driving_log_01, driving_log_02, driving_log_03, driving_log_04, driving_log_05]
df_data = pd.concat(frames)
print(df_data.shape)

train, validation = train_test_split(df_data, test_size=0.2)

# Path correction
current_path = os.getcwd()
print(current_path)
image_path = os.path.join('data', 'IMG')
full_data_path = os.path.join(current_path, image_path)
print(full_data_path)


def generator(samples, full_data_path, batch_size=128):
    num_samples = len(samples)
    print('Number of training samples: {}'.format(num_samples))
    print('Number of augmented training samples: {}'.format(num_samples*6))

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            labels = []
            for batch_sample in batch_samples.iterrows():
                # Center image
                X, y = get_data(batch_sample, 'img_center', full_data_path)
                images.append(X)
                labels.append(y)

                # Center image flipped
                X, y = get_data(batch_sample, 'img_center', full_data_path, flipped=True)
                images.append(X)
                labels.append(y)

                # Left image
                X, y = get_data(batch_sample, 'img_left', full_data_path)
                images.append(X)
                labels.append(y)

                # Left image flipped
                X, y = get_data(batch_sample, 'img_left', full_data_path, flipped=True)
                images.append(X)
                labels.append(y)

                # Right image
                X, y = get_data(batch_sample, 'img_right', full_data_path)
                images.append(X)
                labels.append(y)

                # Right image flipped
                X, y = get_data(batch_sample, 'img_right', full_data_path, flipped=True)
                images.append(X)
                labels.append(y)

            X_train = np.array(images)
            y_train = np.array(labels)
            yield shuffle(X_train, y_train)


def get_data(row, img_column, data_path, flipped=False):
    if flipped:
        steering_factor = (-1.)
    else:
        steering_factor = 1.

    delimiters = '/', '\\'
    regex_pattern = '|'.join(map(re.escape, delimiters))
    file = re.split(regex_pattern, row[1][img_column])[-1]
    # Correct path
    path = os.path.join(data_path, file)
    image = cv2.imread(path)

    if flipped:
        image = cv2.flip(image, 1)

    steering = steering_factor * row[1]['steering']

    # Steering correction for non center images
    if img_column != 'img_center':
        if img_column == 'img_left':
            steering += steering_factor * steering_correction
            if not flipped:
                steering = min(steering, MAX_STEERING)
            else:
                steering = max(steering, MIN_STEERING)
        elif img_column == 'img_right':
            steering -= steering_factor * steering_correction
            if not flipped:
                steering = max(steering, MIN_STEERING)
            else:
                steering = min(steering, MAX_STEERING)

    throttle = row[1]['throttle']
    brake = row[1]['brake']
    speed = row[1]['speed']

    # Normalize throttle, brake
    if brake != 0:
        acceleration = (-1.) * brake
    else:
        acceleration = 1. * throttle

    # Use this line to predict steering angle and throttle/break
    # label = [steering, acceleration]
    label = [steering]
    return image, label


train_generator = generator(train, full_data_path, batch_size=BATCH_SIZE)
validation_generator = generator(validation, full_data_path, batch_size=BATCH_SIZE)

# Lambda functions
def hsv_conversion(img):
    import tensorflow as tf
    return tf.image.rgb_to_hsv(img)

print('\nStart training\n')
model = Sequential()
model.add(Lambda(hsv_conversion, input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(50, activation='relu'))
# Uncomment this line to predict steering angle and throttle/break
# model.add(Dense(2, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(train)/BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=len(validation)/BATCH_SIZE,
                    epochs=50,
                    verbose=1
                    )

model.save('model.h5')
