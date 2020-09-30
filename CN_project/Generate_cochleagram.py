import pycochleagram.cochleagram as cgram
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import dirname, join as join
from scipy.io.wavfile import read


def load_data(dir):
    filenames = listdir(dir)
    labels = []
    audios = []
    Fs = []
    for i in range(len(filenames)):
        fs, audio = read(join(dir, filenames[i]))
        audios.append(audio)
        label = int(filenames[i][0])  # to get the first character in the string, which gives the digit pronounced
        labels.append(label)
        Fs.append(fs)
    return audios, labels, Fs


def make_cochlea(audio, Fs, labels):
    cochleagrams = []
    idx = range(len(labels))
    for a, f, l, i in zip(audio, Fs, labels, idx):
        coch = cgram.human_cochleagram(a, f, hi_lim=4000) #Nyquist frequency: half of the sampling rate
        np.save(join(outdir, f"{l}_{i}.npy"), coch)


def visualize_coch():
    coch = np.load(join(outdir, '0_0.npy'))
    plt.imshow(coch, aspect='auto', origin='lower')
    plt.show()


if __name__ == '__main__':
    dir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master/recordings"
    outdir = "/home/esther/Documents/Uni/SB/Block7/Computational neuro/project/cochleagrams"
    audio, labels, Fs = load_data(dir)
    make_cochlea(audio, Fs, labels)
    visualize_coch()
    print("hello")