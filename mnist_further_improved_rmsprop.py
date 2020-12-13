# Seite 15 und Seite 16 im Buch 

import tensorflow as tf
import numpy as np
from tensorflow import keras



EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10  
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

mnist = keras.datasets.mnist  
(X_train, Y_train),(X_test,Y_test) = mnist.load_data()

RESHAPED = 784  
X_train = X_train.reshape(60000,RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255  
X_test /= 255  

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test =  tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,input_shape=(RESHAPED,),name='dense_layer',activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN,input_shape=(RESHAPED,),name='dense_layer2',activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(NB_CLASSES,input_shape=(RESHAPED,),name='dense_layer3',activation='softmax'))

model.summary()

model.compile(optimizer='RMSProp',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)


test_loss, test_acc = model.evaluate(X_test,Y_test)
print('Test accuracy:',test_acc)