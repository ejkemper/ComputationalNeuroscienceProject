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
from Classify_latent_space import Classifier, make_dataset as make_classifier_dataset

#For the unimodal model the losses are adjusted. We train the model on one of the inputs all the time and expect it to return the input.
#We do not care about the output of the other steam, so 2 seperate loss functions are needed.

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
            keras.layers.Reshape((28, 28))
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


def seperate_loss(y_true, y_pred):
    MSE = tf.square(y_true - y_pred)
    MSE = tf.reduce_mean(MSE, axis=[1, 2])
    is_input = y_true != 0
    importance = tf.math.reduce_any(is_input, axis=[1, 2])  # indicates whether any pixel in an instance is not equal to zero (meaning that it is a 'real' image
    loss = tf.cast(importance, tf.float32) * MSE #makes sure that only loss is used when its from the desired input. We do not care about output from empty input
    return loss


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


def make_dataset(data_im, labels_im, data_ch, labels_ch):
    data_im_zeros = np.zeros([len(data_im), 15, 53])
    data_ch_zeros = np.zeros([len(data_ch), 28, 28])

    data_1 = np.concatenate([data_im, data_ch_zeros], axis=0)
    data_2 = np.concatenate([data_im_zeros, data_ch], axis=0)

    output_ch = np.zeros([len(data_im) + len(data_ch), 15, 53])
    output_im = np.zeros([len(data_im) + len(data_ch), 28, 28])
    for i in range(len(data_im)):
        #output_ch are the cochleagrams that we want to have as ouput when an image is given
        #output_im are the images that we want as output when a cochleagram is given
        output_ch[i] = data_ch[np.random.choice(np.where(labels_ch == labels_im[i])[0])]

    for i in range(len(data_im), len(data_ch)+len(data_im)):
        output_im[i] = data_im[np.random.choice(np.where(labels_im == labels_ch[i-len(data_im)])[0])]

    data = [data_1, data_2]
    outputs = [output_im, output_ch]
    return data, outputs


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output_crossmodal_AE"
    data_coch, labels_coch = load_files(dir)
    train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2,
                                                                                  random_state=42)

    mnist = keras.datasets.mnist
    (train_data_mnist, train_label_mnist), (test_data_mnist, test_label_mnist) = mnist.load_data()
    train_data_mnist = train_data_mnist / 255.0
    test_data_mnist = test_data_mnist / 255.0

    train_data, train_outputs = make_dataset(train_data_mnist, train_label_mnist, train_data_ch, train_label_ch)
    test_data, test_outputs = make_dataset(test_data_mnist, test_label_mnist, test_data_ch, test_label_ch)

    model = Autoencoder(latent_dim=64)

    model.compile(optimizer='adam',
                  loss=seperate_loss)

    model.fit(train_data, train_outputs, epochs=50, shuffle=True, validation_data=(test_data, test_outputs))
    test_loss = model.evaluate(test_data, test_outputs, verbose=2)

    save_dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/models/crossmodal"
    model.save(save_dir)

    #Check results
    #image from cochleagram as input
    rep_im, _ = model.call([np.zeros([1, 28, 28]), test_data[1][10100:10101]])
    #cochleagram from image as input
    _, rep_ch = model.call([test_data[0][1:2], np.zeros([1, 15, 53])])

    rep_ch = rep_ch.numpy()
    rep_im = rep_im.numpy()

    # plt.imshow(test_data[0][1])
    # plt.figure()
    # plt.imshow(rep_im.reshape((28, 28)))
    # plt.figure()
    #
    # plt.imshow(test_data[1][10100])
    # plt.figure()
    # plt.imshow(rep_ch.reshape((15, 53)))
    # plt.show()

    train_data, train_labels = make_classifier_dataset(train_data_mnist, train_data_ch)
    test_data, test_labels = make_classifier_dataset(test_data_mnist, test_data_ch)
    classifier = Classifier(model)
    classifier.compile(optimizer='adam',
                       loss=keras.losses.BinaryCrossentropy(from_logits=True),
                       # since input are values, no probabilities
                       metrics=['accuracy'])

    classifier.fit(train_data, train_labels, epochs=5, shuffle=True, validation_data=(test_data, test_labels))
    test_loss, test_acc = classifier.evaluate(test_data, test_labels, verbose=2)