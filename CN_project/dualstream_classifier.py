import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat, savemat
from os import listdir
from os.path import dirname, join as join

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder1_im = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),     #input
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.encoder1_ch = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15, 53)),
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.decoder = tf.keras.Sequential([

        ])
    def call(self, im, ch):
        encoded_im = self.encoder1_im(im)
        encoded_ch = self.encoder1_ch(ch)
        decoded = self.decoder()
        return decoded

def load_data(dir):
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



if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    data, labels = load_data(dir)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = Autoencoder(latent_dim=64)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())

    model.fit(train_data, train_label, epochs=50, shuffle=True, validation_data=(test_data, test_label))
    test_loss = model.evaluate(test_data, test_label, verbose=2)
