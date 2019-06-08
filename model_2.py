import glob
import os

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from pandas.tests.extension.numpy_.test_numpy_nested import np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.data import Dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

# Define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# Variables to be tuned
W = tf.Variable (tf.truncated_normal ([image_size * image_size, labels_size], stddev=0.1))
b = tf.Variable (tf.constant (0.1, shape=[labels_size]))

# Build the network (only output layer)
output = tf.matmul (training_data, W) + b

# Define the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

# Training step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Accuracy calculation
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

