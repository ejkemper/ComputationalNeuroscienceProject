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


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        activation = 'relu'
        self.encoder1_im = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # input
            keras.layers.Dense(32, activation=activation)
        ])

        self.encoder1_ch = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15, 53)),
            keras.layers.Dense(32, activation=activation)
        ])

        self.combine = tf.keras.Sequential([
            keras.layers.Concatenate(axis=-1),
            keras.layers.Dense(64, activation=activation)  # latent space
        ])

        self.decoder_im = tf.keras.Sequential([
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(784, activation=activation),
            keras.layers.Reshape((28,28))
        ])

        self.decoder_ch = tf.keras.Sequential([
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(15 * 53, activation=activation),
            keras.layers.Reshape((15, 53))
        ])

    def encode(self, im, ch):
        encoded_im = self.encoder1_im(im)
        encoded_ch = self.encoder1_ch(ch)
        combined_encoding = self.combine([encoded_im, encoded_ch])
        return combined_encoding

    def decode(self, encoding):
        decoded_im = self.decoder_im(encoding)
        decoded_ch = self.decoder_ch(encoding)
        return decoded_im, decoded_ch

    def call(self, input):
        im, ch = input
        encoding = self.encode(im, ch)
        decoded_im, decoded_coch = self.decode(encoding)
        return decoded_im, decoded_coch


def load_files(dir):
    filenames = listdir(dir)
    files = np.zeros((len(filenames), 15, 53))
    labels = []
    for i in range(len(filenames)):
        file = np.load(join(dir, filenames[i]))
        files[i] = file
        label = int(filenames[i][0])  # to get the 6th character in the string, which gives the digit pronounced
        labels.append(label)
    labels = np.array(labels)
    return files, labels


# this function always needs longest input first!
def match_data(data1, labels1, data2, labels2):
    data_results_1 = []
    data_results_2 = []
    labels = []
    for x in range(10):
        nr_data1 = len(labels1[labels1 == x])
        nr_data2 = len(labels2[labels2 == x])
        reps = np.ceil(nr_data1 / nr_data2)

        rep_data_2 = np.repeat(data2[labels2 == x], reps, axis=0)
        rep_data_2 = rep_data_2[:nr_data1]
        data_results_2.append(rep_data_2)

        data_results_1.append(data1[labels1 == x])
        labels.append(labels1[labels1 == x])

    data_results_1 = np.concatenate(data_results_1, axis=0)
    data_results_2 = np.concatenate(data_results_2, axis=0)
    data = [data_results_1, data_results_2]
    labels = np.concatenate(labels, axis=0)
    return data, labels


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    data_coch, labels_coch = load_files(dir)
    train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2,
                                                                                  random_state=42)

    mnist = keras.datasets.mnist
    (train_data_mnist, train_label_mnist), (test_data_mnist, test_label_mnist) = mnist.load_data()
    train_data_mnist = train_data_mnist / 255.0
    test_data_mnist = test_data_mnist / 255.0

    # train_data, train_labels = make_dataset(train_data_mnist, train_label_mnist, train_data_ch, train_label_ch)

    train_data, train_labels = match_data(train_data_mnist, train_label_mnist, train_data_ch, train_label_ch)
    test_data, test_label = match_data(test_data_mnist, test_label_mnist, test_data_ch, test_label_ch)

    model = Autoencoder(latent_dim=64)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())

    model.fit(train_data, train_data, epochs=50, shuffle=True, validation_data=(test_data, test_data))
    test_loss = model.evaluate(test_data, test_data, verbose=2)

    rep_im, rep_ch = model.call([test_data[0][1:2], test_data[1][1:2]])
    rep_im = rep_im.numpy()
    rep_ch = rep_ch.numpy()
    plt.imshow(test_data[0][1])
    plt.figure()
    plt.imshow(rep_im.reshape((28, 28)))
    plt.figure()
    plt.imshow(test_data[1][1])
    plt.figure()
    plt.imshow(rep_ch.reshape((15, 53)))
    plt.show()