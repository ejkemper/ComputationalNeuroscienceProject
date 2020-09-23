import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat, savemat
from os import listdir
from os.path import dirname, join as join

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print(tf.__version__)

#load data
def load_data(dir):
    filenames = listdir(dir)
    files = np.zeros((len(filenames), 23, 178))
    labels = []
    for i in range(len(filenames)):
        file = loadmat(join(dir, filenames[i]))
        file = normalize_data(file['coch'])
        files[i] = file
        label = int(filenames[i][6]) #to get the 6th character in the string, which gives the digit pronounced
        labels.append(label)
    labels = np.array(labels)
    return files, labels


#normalize input
def normalize_data(file):
    file = (file - np.min(file)) / (np.max(file) - np.min(file))
    return file

# all_loss = params.history['loss']
# all_acc = params.history['accuracy']
# plt.plot(all_loss, label='loss')
# plt.plot(all_acc, label='accuracy')
# plt.legend()
# plt.show()

if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    data, labels = load_data(dir)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(23, 178)),
        keras.layers.Dense(16, activation='sigmoid'),
        keras.layers.Dense(16, activation='sigmoid'),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    params = model.fit(train_data, train_label, epochs=100, shuffle=True)
    test_loss, test_acc = model.evaluate(test_data, test_label, verbose=2)
    print('test acc:', test_acc)

    print('help')