import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = tf.keras.datasets.mnist

img_rows, img_cols = 28, 28

# the data, split between train and test sets

batch_size = 1000
num_classes = 10
epochs = 1

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout

def build_model():
    model = Sequential()
    model.add(Conv2D(8, 9, strides=(4, 4), padding="same", activation="elu", input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(16, 5, strides=(2, 2), padding="same", activation="elu"))
    model.add(Conv2D(32, 4, strides=(1, 1), padding="same", activation="elu"))
    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(2000, activation="elu"))
    model.add(Dense(1000, activation="elu"))
    model.add(Dropout(.3))
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

'''model.fit_generator(
        generator=mnist.train.next_batch(1000),
        steps_per_epoch=mnist.train.images.shape[0] / 1000,
        validation_data=mnist.validation.next_batch(1000),
        validation_steps=mnist.validation.images.shape[0] / 1000,
        epochs=10)'''