import numpy as np
import matplotlib
from os import listdir
from os.path import dirname, join as join
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


def load_filenames(dir):
    filenames = listdir(dir)
    return filenames

def zero_pad(filenames, dir, outdir):
    longest_file = loadmat(join(dir, "digit_9_209"))
    length = longest_file["coch"].shape[1]

    for f in filenames:
        file = loadmat(join(dir, f))
        curr_length = file["coch"].shape[1]
        pad_length = length - curr_length
        rand = np.random.random()
        pad_left = int(np.floor((pad_length/2) * rand))
        pad_right = int(pad_length - pad_left)
        print(curr_length, 'file', f)
        new_file = {'coch' : np.pad(file["coch"], [[0,0], [pad_left, pad_right]])}
        savemat(join(outdir, f), new_file, do_compression = True)

def downsample():


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/cochleagrams"
    outdir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/output"
    filenames = load_filenames(dir)
    zero_pad(filenames, dir, outdir)
    print('test')