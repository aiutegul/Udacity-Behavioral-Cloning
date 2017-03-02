import csv
import cv2

def crop_image(image):
	return image[:, 60:136, :, :]

lines = [] 
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
#read images and measurements from the csv file
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = "./data/IMG/" + filename
		image = cv2.imread(local_path)
		rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(rgbImage)
	correction = 0.15
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement + correction)
	measurements.append(measurement - correction)

#augmenting images by flipping them, and measurements by multiplying by -1
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

import numpy as np

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#NVDIA's architecture with an additional layer for cropping images
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout 
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(crop_image, input_shape = (160, 320, 3)))
model.add(Lambda (lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')


