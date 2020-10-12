import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np



# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)), #input layer
#     keras.layers.Dense(32, activation='sigmoid'), #encoder
#     keras.layers.Dense(784, activation='sigmoid'), #decoder
#     keras.layers.Reshape((28,28))
# ])

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),             # input layer
            keras.layers.Dense(latent_dim, activation='sigmoid')    # encoder
        ])
        self.decoder = tf.keras.Sequential([
            keras.layers.Dense(784, activation='sigmoid'),      # decoder
            keras.layers.Reshape((28, 28))                      # makes vector to matrix (image)
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':

    mnist = keras.datasets.mnist
    (train_imgs, _), (test_imgs, _) = mnist.load_data()

    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0

    model = Autoencoder(latent_dim=64)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError())

    model.fit(train_imgs, train_imgs, epochs=50, shuffle=True, validation_data=(test_imgs, test_imgs))
    test_loss = model.evaluate(test_imgs,  test_imgs, verbose=2)

    test_im = test_imgs[1:8]
    encoded_imgs = model.encoder(test_im).numpy()
    decoded_imgs = model.decoder(encoded_imgs).numpy()

    for i in range(len(test_im)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(test_im[i])
        ax2.imshow(decoded_imgs[i])
    plt.show()

    plt.imshow(test_im[0])

    print("end")

