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

        self.encoder1_im = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),     #input
            keras.layers.Dense(32, activation='sigmoid')
        ])
        self.encoder2_im = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),  # input
            keras.layers.Dense(32, activation='sigmoid')
        ])

        self.encoder1_ch = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15, 53)),
            keras.layers.Dense(32, activation='sigmoid')
        ])
        self.encoder2_ch = tf.keras.Sequential([
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

def load_files(dir):
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


def make_dataset(mnist_data, mnist_labels, coch_data, coch_labels):
    #Here we want to store all the labels from all cases together.
    labels = []

    #first we want to have case 1. We have 2 images that we want to use as input with the same label.
    #We want the cochleagrams to be empty
    rand_im, rand_lab = rand_per_label(mnist_data, mnist_labels)
    data_im_im, labels_im_im = match_data(mnist_data, mnist_labels, rand_im, rand_lab)
    #make empty inputs for the cochleagrams
    data_coch_zeros = np.zeros([len(data_im_im[0]), 15, 53])
    #since we want to append two empty cochleagrams for this combination
    data_im_im.append(data_coch_zeros)
    data_im_im.append(data_coch_zeros)
    labels.append(labels_im_im)

    #Then case 2: we have 2 cochleagrams as input and want to append 2 empty images for this set
    #rest same as above
    rand_coch, rand_lab = rand_per_label(coch_data, coch_labels)
    data_ch_ch, labels_ch_ch = match_data(coch_data, coch_labels, rand_coch, rand_lab)
    data_im_zeros = np.zeros([len(data_ch_ch[0]), 28, 28])
    data_ch_ch.insert(0, data_im_zeros)
    data_ch_ch.insert(0, data_im_zeros)
    labels.append(labels_ch_ch)

    #Then case 3: Now we want to combine the cochleagrams and the images, so we do not have to randomize the orders
    #But we also need to randomize to which input it is given
    data_im_ch, labels_im_ch = match_data(mnist_data, mnist_labels, coch_data, coch_labels)
    #We want to add an empty input for one of the ims and for one of the cochs
    data_coch_zeros = np.zeros([len(data_im_ch[0]), 15, 53])
    data_im_zeros = np.zeros([len(data_im_ch[0]), 28, 28])
    #To randomly select indices from the list, so we can set those to input 0 and others to 1 for ims
    #For cochs the same but then input 2 and 3
    idx = range(len(data_im_ch[0]))
    ims_to_move = np.random.choice(idx, int(np.ceil(len(idx)/2)), replace=False)
    ims_to_keep = np.array(list(set(idx) - set(ims_to_move))) # Indices that are not selected
    #make 2 image inputs to split them according to rand choice above
    imgs_1 = data_im_zeros.copy() # copy is needed otherwise imgs_2 would also be changed when we change this array
    imgs_2 = data_im_zeros
    imgs_1[ims_to_move] = data_im_ch[0][ims_to_move]  # Copies the selected images to input0
    imgs_2[ims_to_keep] = data_im_ch[0][ims_to_keep]  # Copies the rest to input1

    # Then we do the same for the cochleagrams
    cochs_to_move = np.random.choice(idx, int(np.ceil(len(idx)/2)), replace=False)
    cochs_to_keep = np.array(list(set(idx) - set(cochs_to_move)))
    cochs_1 = data_coch_zeros.copy()
    cochs_2 = data_coch_zeros
    cochs_1[cochs_to_move] = data_im_ch[1][cochs_to_move]  # Copies the selected cochs to input2
    cochs_2[cochs_to_keep] = data_im_ch[1][cochs_to_keep]  # Copies the rest to input3

    # data_im_ch can now be these new arrays
    data_im_ch = [imgs_1, imgs_2, cochs_1, cochs_2]
    labels.append(labels_im_ch)

    del rand_im
    del rand_coch
    del rand_lab

    #postprocess output
    labels = np.concatenate(labels, axis=0)
    data = [np.concatenate([data_im_im[0], data_ch_ch[0], data_im_ch[0]]),
            np.concatenate([data_im_im[1], data_ch_ch[1], data_im_ch[1]]),
            np.concatenate([data_im_im[2], data_ch_ch[2], data_im_ch[2]]),
            np.concatenate([data_im_im[3], data_ch_ch[3], data_im_ch[3]]),
            ]

    return data, labels

#this function always needs longest input first!
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


def rand_per_label(data, labels):
    rand_data = []
    labs = []
    for x in range(10):
        data_x = data[labels == x]
        np.random.shuffle(data_x)
        rand_data.append(data_x)
        l = np.repeat(x, len(data_x))
        labs.append(l)
    #to fix the axis and make one big array
    rand_data = np.concatenate(rand_data, axis=0)
    labs = np.concatenate(labs, axis=0)
    return rand_data, labs

if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    data_coch, labels_coch = load_files(dir)
    train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2, random_state=42)

    mnist = keras.datasets.mnist
    (train_data_mnist, train_label_mnist), (test_data_mnist, test_label_mnist) = mnist.load_data()
    train_data_mnist = train_data_mnist / 255.0
    test_data_mnist = test_data_mnist / 255.0

    train_data, train_labels = make_dataset(train_data_mnist, train_label_mnist, train_data_ch, train_label_ch)

    model = Autoencoder(latent_dim=64)

    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.MeanSquaredError())
    #
    # model.fit(train_data, train_data, epochs=50, shuffle=True, validation_data=(test_data, test_data))
    # test_loss = model.evaluate(test_data, test_data, verbose=2)
