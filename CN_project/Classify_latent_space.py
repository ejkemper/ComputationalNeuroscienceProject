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


class Classifier(Model):
    def __init__(self, model):
        super(Classifier, self).__init__()
        # because we have a binary classifier, sigmoid is more suitable
        activation = 'sigmoid'
        # self.model: keras.Model = tf.keras.models.load_model(dir, custom_objects={"seperate_loss": seperate_loss})
        self.model = model
        for layer in self.model.layers:
            layer.trainable = False

        self.classifier = tf.keras.Sequential([
            keras.layers.Dense(1, activation=activation)  # latent space
        ])

    def call(self, input):
        im, ch = input
        latent_space = self.model.encode(im, ch)
        classified = self.classifier(latent_space)
        return classified


def make_dataset(data_im, data_ch):
    data_im_zeros = np.zeros([len(data_im), 15, 53])
    data_ch_zeros = np.zeros([len(data_ch), 28, 28])

    data_1 = np.concatenate([data_im, data_ch_zeros], axis=0)
    data_2 = np.concatenate([data_im_zeros, data_ch], axis=0)

    #if input is an image, output 1, if input is a cochleagram output 0
    im_lab = np.ones(len(data_im))
    ch_lab = np.zeros(len(data_ch))
    labels = np.concatenate([im_lab, ch_lab])

    data = [data_1, data_2]
    return data, labels


if __name__ == '__main__':
    from dualstream_autoencoder_unimodal import load_files
    dir_unimodal = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/models/unimodal"
    dir_bimodal = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/models/bimodal"
    dir_crossmodal = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/models/crossmodal"

    dir_data = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    dir_data_crossmodal = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output_crossmodal_AE"

    data_coch, labels_coch = load_files(dir_data)
    train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2,
                                                                                  random_state=42)
    mnist = keras.datasets.mnist
    (train_data_mnist, train_label_mnist), (test_data_mnist, test_label_mnist) = mnist.load_data()
    train_data_mnist = train_data_mnist / 255.0
    test_data_mnist = test_data_mnist / 255.0

    train_data, train_labels = make_dataset(train_data_mnist, train_data_ch)
    test_data, test_labels = make_dataset(test_data_mnist, test_data_ch)

    model = Classifier(dir_unimodal)

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True), #since input are values, no probabilities
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=50, shuffle=True, validation_data=(test_data, test_labels))
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)