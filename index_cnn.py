import os,re
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras.models import Model
from keras import Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255

from keras.utils import to_categorical 
y_train = to_categorical(labels_train, 10) 
y_test = to_categorical(labels_test, 10)

x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

from keras.layers import Dropout
from keras.models import Sequential
from keras import regularizers
from keras import initializers

model = Sequential()
model_drop = Sequential()
model_drop.add(Dense(512, activation='relu', input_shape=(28,28,1), kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.0625, seed=None), kernel_regularizer=regularizers.l2(0.0001)))
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))
model_drop.add(Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.088, seed=None), kernel_regularizer=regularizers.l2(0.0001)))
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))
model_drop.add(Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.125, seed=None),kernel_regularizer=regularizers.l2(0.0001)) )
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))
model_drop.add(Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1767, seed=None),kernel_regularizer=regularizers.l2(0.0001)) )
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))
model_drop.add(Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.25, seed=None),kernel_regularizer=regularizers.l2(0.0001)) )
model_drop.add(BatchNormalization())
model_drop.add(Dropout(0.5))
model_drop.add(Flatten())
model_drop.add(Dense(10, activation='softmax'))
model_drop.summary()

batch_size = 128
num_classes = 10
epochs = 20

model_drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_drop.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['loss'], label='training loss') 
plt.plot(history.history['val_loss'], label='validation loss') 
plt.xlabel('epochs')
plt.ylabel('loss') 
plt.legend()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("network_for_mnist.h5")

from keras.models import load_model 
net=load_model("network_for_mnist.h5")

outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1) 
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)
net=load_model("network_for_mnist.h5")

outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1) 
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)

plt.figure(figsize=(8, 2)) 
for i in range(0,8):
  ax=plt.subplot(2,8,i+1)
  plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r')) 
  plt.title(labels_test[i])
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False)

for i in range(0,8):
  #output = net.predict(x_test[i,:].reshape(1, 784)) #if MLP 
  output = net.predict(x_test[i,:].reshape(1, 28,28,1)) #if CNN 
  output=output[0,0:]
  plt.subplot(2,8,8+i+1)
  plt.bar(np.arange(10.),output)
  plt.title(np.argmax(output))
