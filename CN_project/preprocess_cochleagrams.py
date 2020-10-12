import numpy as np
import matplotlib
from os import listdir
from os.path import dirname, join as join
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import cv2


def load_filenames(dir):
    filenames = listdir(dir)
    return filenames


def zero_pad(filenames, dir, outdir):
    longest_file = np.load(join(dir, "9_2908.npy"))
    length = longest_file.shape[1]

    for f in filenames:
        file = np.load(join(dir, f))

        curr_length = file.shape[1]
        pad_length = length - curr_length
        rand = np.random.random()
        pad_left = int(np.floor((pad_length / 2) * rand))
        pad_right = int(pad_length - pad_left)
        new_coch = np.pad(file, [[0, 0], [pad_left, pad_right]])

        downs_coch = downsample(new_coch)
        downs_coch = normalize_data(downs_coch)
        # print(new_file['coch'].shape)
        np.save(join(outdir, f), downs_coch)


def normalize_data(file):
    file = (file - np.min(file)) / (np.max(file) - np.min(file))
    return file


def downsample(file):
    image = cv2.resize(file, dsize=(53, 15))
    return image


def visualize_coch(outdir):
    coch = np.load(join(outdir, '0_0.npy'))
    plt.imshow(coch, aspect='auto', origin='lower')
    plt.show()


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/cochleagrams"
    outdir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    filenames = load_filenames(dir)
    zero_pad(filenames, dir, outdir)
    visualize_coch(outdir)
    print('test')
