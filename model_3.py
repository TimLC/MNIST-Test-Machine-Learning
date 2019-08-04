import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

mnist = tf.keras.datasets.mnist

img_rows, img_cols = 28, 28

# the data, split between train and test sets

batch_size = 1000
num_classes = 10
epochs = 100

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout, MaxPooling2D

def build_model():
    model = Sequential()

    model.add(Conv2D(64, 2, strides=(1, 1), padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Conv2D(128, 2, strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Conv2D(256, 2, strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(.6))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True,
                 use_random_shift=True, use_random_zoom=True):
    augmented_image = []
    augmented_image_labels = []

    for num in range(0, dataset.shape[0]):

        for i in range(0, augementation_factor):
            # original image:
            augmented_image.append(dataset[num])
            augmented_image_labels.append(dataset_labels[num])

            if use_random_rotation:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1,
                                                                         channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shear:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1,
                                                                      channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_shift:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1,
                                                                      channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

            if use_random_zoom:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.9, 0.9), row_axis=0, col_axis=1,
                                                                     channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])

    return np.array(augmented_image), np.array(augmented_image_labels)

model = build_model()

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print(x_train.shape)
x_train, y_train = augment_data(x_train, y_train)
print(x_train.shape)
x_train, y_train = shuffle(x_train, y_train)

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

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


test_predictions = model.predict_classes(x_test)
test_p = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_p, test_predictions)

import seaborn as sns

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()
print(history.history['val_acc'])

def print_image():
    for i in range (0,10):
        print(y_train_v2[i]) # The label is 8
        plt.imshow(x_train_v2[i].reshape(28,28), cmap='Greys')
        plt.show()