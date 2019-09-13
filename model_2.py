import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.layers.normalization import BatchNormalization
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = tf.keras.datasets.mnist

img_rows, img_cols = 28, 28

# the data, split between train and test sets

batch_size = 1000
num_classes = 10
epochs = 100

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout, MaxPooling2D

def build_model():
    model = Sequential()

    model.add(Conv2D(32, 2, strides=(1, 1), padding="same", activation="relu", input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(.6))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

model = build_model()

'''def data_set():
    input_batch, labels_batch = mnist.train.next_batch (batch_size)
    input_batch = input_batch.reshape(input_batch.shape[0], img_rows, img_cols, 1)
    input_batch = input_batch.astype('float32')
    input_batch /= 255
    labels_batch = tf.keras.utils.to_categorical (labels_batch, 10)
    print(input_batch.shape)
    return input_batch, labels_batch

def validation_set():
    input_batch, labels_batch = mnist.validation.next_batch(batch_size)
    input_batch = input_batch.reshape(input_batch.shape[0], img_rows, img_cols, 1)
    input_batch = input_batch.astype('float32')
    input_batch /= 255
    labels_batch = tf.keras.utils.to_categorical(labels_batch, 10)
    print(input_batch.shape)
    return input_batch, labels_batch'''

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()

test_predictions = model.predict_classes(x_test)
test_p = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_p, test_predictions)

import seaborn as sns
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()