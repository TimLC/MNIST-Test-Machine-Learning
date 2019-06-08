import os
from urllib.request import urlretrieve
import gzip
import tensorflow as tf

URL = "http://yann.lecun.com/exdb/mnist/"
TRAIN = "./train/"
TEST = "./test/"
URL_DOC_TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
URL_DOC_TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
URL_DOC_TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
URL_DOC_TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

NAME_DOC_TRAIN_IMAGES = "doc-train-images"
NAME_DOC_TRAIN_LABELS = "doc-train-labels"
NAME_DOC_TEST_IMAGES = "doc-test-images"
NAME_DOC_TEST_LABELS = "doc-test-labels"

def repository(repository):
    if not os.path.exists(repository):
        #os.makedirs(repository)
        os.makedirs("./test/")

def download(path, name):
    if not os.path.exists(name):
        urlretrieve(path, name)

def extract(path, data):
    if not os.path.exists(data + path):
        input = gzip.GzipFile(path, 'rb')
        s = input.read()
        input.close()

        output = open(data + path, 'wb')
        output.write(s)
        output.close()

def init_dataset():

    repository(TRAIN)
    repository(TEST)

    download(URL + URL_DOC_TRAIN_IMAGES, NAME_DOC_TRAIN_IMAGES)
    download(URL + URL_DOC_TRAIN_LABELS, NAME_DOC_TRAIN_LABELS)

    download(URL + URL_DOC_TEST_IMAGES, NAME_DOC_TEST_IMAGES)
    download(URL + URL_DOC_TEST_LABELS, NAME_DOC_TEST_LABELS)

    extract(NAME_DOC_TRAIN_IMAGES, TRAIN)
    extract(NAME_DOC_TRAIN_LABELS, TRAIN)
    extract(NAME_DOC_TEST_IMAGES, TEST)
    extract(NAME_DOC_TEST_LABELS, TEST)


    local_train_images_file = os.path.join(TRAIN, NAME_DOC_TRAIN_IMAGES)

    local_train_labels_file = os.path.join(TRAIN, NAME_DOC_TRAIN_LABELS)

    local_test_images_file = os.path.join(TEST, NAME_DOC_TRAIN_IMAGES)

    local_test_labels_file = os.path.join(TEST, NAME_DOC_TRAIN_LABELS)

    return local_train_images_file, local_train_labels_file, local_test_images_file, local_test_labels_file

def read_label(tf_bytestring):
    label = tf.decode_raw(tf_bytestring, tf.uint8)
    return tf.reshape(label, [])

def read_image(tf_bytestring):
    image = tf.decode_raw(tf_bytestring, tf.uint8)
    return tf.cast(image, tf.float32)/256.0

def load_dataset(image_file, label_file):
    image_dataset = tf.data.FixedLengthRecordDataset(image_file, 28*28, header_bytes=16, buffer_size=1024*16).map(read_image)
    labels_dataset = tf.data.FixedLengthRecordDataset(label_file, 1,header_bytes=8, buffer_size=1024*16).map(read_label)
    dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))
    return dataset

def nodes_for_model(dataset):
        features, labels = dataset.make_one_shot_iterator().get_next()
        return {'image': features}, labels

def train_data_input_fn(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(100)
    return nodes_for_model(dataset)

def eval_data_input_fn(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.batch(10000)  # a single batch with all the test data
    dataset = dataset.repeat(1)
    return nodes_for_model(dataset)

download(URL + URL_DOC_TRAIN_IMAGES, NAME_DOC_TRAIN_IMAGES)
download(URL + URL_DOC_TRAIN_LABELS, NAME_DOC_TRAIN_LABELS)

extract(NAME_DOC_TRAIN_IMAGES, TRAIN)
extract(NAME_DOC_TRAIN_LABELS, TRAIN)
