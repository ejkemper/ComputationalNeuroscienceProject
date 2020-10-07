import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat, savemat
from os import listdir
from os.path import dirname, join as join

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Model


class DualStreamClassifier(Model):
    def __init__(self):
        super(DualStreamClassifier, self).__init__()

        self.encoder1_im = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),     #input im
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.encoder1_ch = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15, 53)),     #input coch
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.combine = tf.keras.Sequential([
            keras.layers.Concatenate(axis=-1),
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.classify = tf.keras.Sequential([
            keras.layers.Dense(10, activation='sigmoid')
        ])


    def call(self, input):
        im, ch = input
        encoded_im = self.encoder1_im(im)
        encoded_ch = self.encoder1_ch(ch)
        combined = self.combine([encoded_im, encoded_ch])
        classified = self.classify(combined)
        return classified


def load_files(dir): #coch=15,53,
    filenames = listdir(dir)
    files = np.zeros((len(filenames), 15, 53))
    labels = []
    for i in range(len(filenames)):
        file = np.load(join(dir, filenames[i]))
        files[i] = file
        label = int(filenames[i][0]) #to get the 6th character in the string, which gives the digit pronounced
        labels.append(label)
    labels = np.array(labels)
    return files, labels


def match_data(data_mnist, labels_mnist, data_coch, labels_coch):
    data_ch = []
    data_m = []
    labels = []
    for x in range(10):
        nr_mnist = len(labels_mnist[labels_mnist == x])
        nr_coch = len(labels_coch[labels_coch == x])
        reps = np.ceil(nr_mnist/nr_coch)

        rep_data_ch = np.repeat(data_coch[labels_coch == x], reps, axis=0)
        rep_data_ch = rep_data_ch[:nr_mnist]
        data_ch.append(rep_data_ch)

        data_m.append(data_mnist[labels_mnist == x])
        labels.append(labels_mnist[labels_mnist == x])

    data_ch = np.concatenate(data_ch, axis=0)
    data_m = np.concatenate(data_m, axis=0)
    data = [data_m, data_ch]
    labels = np.concatenate(labels, axis=0)
    return data, labels


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    data_coch, labels_coch = load_files(dir)
    train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2, random_state=42)

    mnist = keras.datasets.mnist
    (train_data_mnist, train_label_mnist), (test_data_mnist, test_label_mnist) = mnist.load_data()
    train_data_mnist = train_data_mnist / 255.0
    test_data_mnist = test_data_mnist / 255.0

    model = DualStreamClassifier()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    train_data, train_label = match_data(train_data_mnist, train_label_mnist, train_data_ch, train_label_ch)
    test_data, test_label = match_data(test_data_mnist, test_label_mnist, test_data_ch, test_label_ch)

    model.fit(train_data, train_label, epochs=50, shuffle=True, validation_data=(test_data, test_label))
    test_loss = model.evaluate(test_data, test_label, verbose=2)
