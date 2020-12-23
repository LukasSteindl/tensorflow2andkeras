import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras import layers, models 


def VGG_16 (weight_path=None):
    model = models.Sequential()
    model.add(layers.ZeroPadding2D((1,1),input_shape=(244,244,3)))
    model.add(layers.Convolution2D(64,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(128,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(256,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(256,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(256,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000,activation='softmax'))
    if weight_path:
        model.load_weights(weight_path)
    return model

##download weights here:
#https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
path_file = os.path.join(os.getcwd(), "weights", "vgg16_weights_tf_dim_ordering_tf_kernels.h5")
model = VGG_16(path_file)
model.summary()
model.compile(optimizer='sgd',loss='categorical_crossentropy')
im = cv2.resize(cv2.imread('elephant.jpg'),(224,224))
im = np.expand_dims(im,axis=0)
out = model.predict(im)
print(np.argmax(out))
print(out)