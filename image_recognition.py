import os
import tensorflow as tf
from tensorflow import keras

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
from keras.utils import np_utils

#print(tf.version.VERSION)
#Looking through training data
#cifar10_class_names = {
#    0: 'Plane',
#    1: 'Car',
#    2: 'Bird',
#    3: 'Cat',
#    4: 'Deer',
#    5: 'Dog',
#    6: 'Frog',
#    7: 'Horse',
#    8: 'Boat',
#    9: 'Truck'
#}

#load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#for i in range(1000): #loop through each pic in data set
#    sample_image = x_train[i] #sample image
#    image_class_number = y_train[i][0] # grab image expected class ID
#    image_class_name = cifar10_class_names[image_class_number] #look up class name from class ID#

#plt.imshow(sample_image) #draw sample image as a plot
#plt.title(image_class_name) #label image
#plt.show()

#normalize data to 0-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#convert class vectors to binary class matrices
#our labels are single values from 0-9
#Instead we want each label to be an array with one element set to 1 and the rest set to 0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Simple neural network
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #will divide image ito 2x2 squares and only take largest value from each region
model.add(Dropout(0.25)) #drop 25% of data
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) #add pooling layers after convolutional layers
model.add(Dropout(0.25))
model.add(Flatten()) #flatten layers tell keras we are no longer working with 2d data
model.add(Dense(512, activation='relu')) #input dim specifies number of nodes in input param
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #activation functions decide inputs from previous layer feeding to next layer. 'relu'= rectified linear function
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
) #optimizer alg used to train nn. 

#model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

#Saving neural network structure - save structure separately from the weights
model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)

#saving weights
model.save_weights('model_weights.h5')





