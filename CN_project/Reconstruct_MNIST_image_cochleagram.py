import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
from os import listdir
from scipy.io import wavfile

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),     # input layer
            keras.layers.Dense(latent_dim, activation='sigmoid')    # encoder
        ])
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(784, activation='sigmoid'),  # decoder
            keras.layers.Reshape((28, 28))                   # makes vector to matrix (image)
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_data():
    dir = '/home/esther/Documents/Uni/SB/Project2/data_numeric_spoken/free-spoken-digit-dataset-master/recordings/'
    files = listdir(dir)
    audiofiles = []
    for i in range(len(files)):
        sr, audio = wavfile.read(dir+files[i])
        audiofiles.append(audio)
    return audiofiles



if __name__ == '__main__':
    wav_files = load_data()
    print('Hello')

