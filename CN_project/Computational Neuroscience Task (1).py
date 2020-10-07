#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Generate cochleagrams - ony needs to be run once on your pc!
#import packages
import pycochleagram.cochleagram as cgram
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import dirname, join as join
from scipy.io.wavfile import read

#load data from dir
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

#make cochleagrams and savein outdir
def make_cochlea(audio, Fs, labels):
    cochleagrams = []
    idx = range(len(labels))
    for a, f, l, i in zip(audio, Fs, labels, idx):
        coch = cgram.human_cochleagram(a, f, hi_lim=4000) #Nyquist frequency: half of the sampling rate
        np.save(join(outdir, f"{l}_{i}.npy"), coch)

#visualize cochleagrams
def visualize_coch():
    coch = np.load(join(outdir, '0_0.npy'))
    plt.imshow(coch, aspect='auto', origin='lower')
    plt.show()

#run functions defined above; adapt dir and outdir to your pc
if __name__ == '__main__':
    dir = "C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/free-spoken-digit-dataset-master/recordings"
    outdir = "C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/cochleagrams"
    audio, labels, Fs = load_data(dir)
    make_cochlea(audio, Fs, labels)
    visualize_coch()
    print("hello")


# In[1]:


#Preprocess cochleagrams - run only once, after "Generate cochleagrams"
#import packages
import numpy as np
import matplotlib
from os import listdir
from os.path import dirname, join as join
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import cv2

#Load cochleagrams from dir
def load_filenames(dir):
    filenames = listdir(dir)
    return filenames

#Add zeropadding 
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

#Normalize data
def normalize_data(file):
    file = (file - np.min(file)) / (np.max(file) - np.min(file))
    return file

#Resize to 53,15
def downsample(file):
    image = cv2.resize(file, dsize=(53, 15))
    return image

#Visualize prepocessed cochleagram
def visualize_coch(outdir):
    coch = np.load(join(outdir, '0_0.npy'))
    plt.imshow(coch, aspect='auto', origin='lower')
    plt.show()

#Call above defined functions; adapt dir and outdir to your pc
if __name__ == '__main__':
    dir = "C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/cochleagrams"
    outdir = "C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/cochleagrams-preprocessed"
    filenames = load_filenames(dir)
    zero_pad(filenames, dir, outdir)
    visualize_coch(outdir)
    print('test')


# In[48]:


#Import preprocessed data
#Import packages
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat, savemat
from os import listdir
from os.path import dirname, join as join
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print(tf.__version__)


#import visual representation - mnist
mnist =  keras.datasets.mnist
(train_imgs_mnist, train_labs_mnist), (test_imgs_mnist, test_labs_mnist) = mnist.load_data()

train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

#import cochleagrams 
def load_files(dir): #coch=15,53,
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

dir = "C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/cochleagrams-preprocessed"
data_coch, labels_coch = load_files(dir)
train_data_ch, test_data_ch, train_label_ch, test_label_ch = train_test_split(data_coch, labels_coch, test_size=0.2, random_state=42)


# In[7]:


### Task classifier 1 - 2 visual inputs, addition 
train_imgs = train_imgs_mnist
train_labs = train_labs_mnist
test_imgs = test_imgs_mnist
test_labs = test_imgs_mnist

#Make 3 splits for images and labels
train_imgs_1=train_imgs[:20000]
train_imgs_2=train_imgs[20000:40000]
train_imgs_3=train_imgs[40000:]
train_labs_1=train_labs[:20000]
train_labs_2=train_labs[20000:40000]
train_labs_3=train_labs[40000:]

#Both combo's contain all 3 splits
train_combo_1=np.concatenate((train_imgs_1,train_imgs_2,train_imgs_3))
train_combo_2=np.concatenate((train_imgs_2,train_imgs_3,train_imgs_1))
train_labs_combo_1=np.concatenate((train_labs_1,train_labs_2,train_labs_3))
train_labs_combo_2=np.concatenate((train_labs_2,train_labs_3,train_labs_1))

#Addtion
train_labs_addition = train_labs_combo_1+train_labs_combo_2

#Test images
test_imgs_1=test_imgs[:5000]
test_imgs_2=test_imgs[5000:]
test_labs_1=test_labs[:5000]
test_labs_2=test_labs[5000:]
test_labs_addition = test_labs_1+test_labs_2

#Input 1
input1 = keras.layers.Input(shape=(28,28))
flatten1 = keras.layers.Flatten(input_shape=(28,28))(input1)
l1_1 = keras.layers.Dense(32, activation='sigmoid')(flatten1)
l2_1 = keras.layers.Dense(32, activation='sigmoid')(l1_1)
l3_1 = keras.layers.Dense(10, activation='sigmoid')(l2_1)

#Input 2
input2= keras.layers.Input(shape=(28,28))
flatten2 = keras.layers.Flatten(input_shape=(28,28))(input2)
l1_2 = keras.layers.Dense(32, activation='sigmoid')(flatten2)
l2_2 = keras.layers.Dense(32, activation='sigmoid')(l1_2)
l3_2 = keras.layers.Dense(10, activation='sigmoid')(l2_2)

#Concatenation
concatenated = keras.layers.Concatenate()([l3_1, l3_2])
l4 = keras.layers.Dense(32, activation = 'sigmoid')(concatenated)
#Output
out= keras.layers.Dense(19)(l4)

#Model
model_2 = keras.models.Model(inputs=[input1, input2], outputs = out)
model_2.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
params_model_add = model_2.fit([train_combo_1, train_combo_2], train_labs_addition, epochs = 25, shuffle=True)

#Evaluation
test_loss, test_acc = model_2.evaluate([test_imgs_1, test_imgs_2],  test_labs_addition, verbose=2)
print('test acc:', test_acc)

all_loss_task = params_model_add.history['loss']
all_acc_task = params_model_add.history['acc']
plt.plot(all_loss_task, label='loss')
plt.plot(all_acc_task, label='accuracy')
plt.legend()
plt.show()


# In[57]:


#Input - make sure for every number the amount of visual and auditory presentations are the same 
def match_data(data_mnist, labels_mnist, data_coch, labels_coch):
    data_ch = []
    data_m = []
    labels = []
    for x in range(10):
        nr_mnist = len(labels_mnist[labels_mnist == x])
        nr_coch = len(labels_coch[labels_coch == x])
        reps = np.ceil(nr_mnist/nr_coch)

        rep_data_ch = np.repeat(data_coch[labels_coch == x], reps, axis=0)
        rep_data_ch = rep_data_ch[:nr_mnist]
        data_ch.append(rep_data_ch)

        data_m.append(data_mnist[labels_mnist == x])
        labels.append(labels_mnist[labels_mnist == x])

    data_ch = np.concatenate(data_ch, axis=0)
    data_m = np.concatenate(data_m, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data_ch, data_m, labels
#output (created with match_data)
train_data_cochleagrams, train_data_visual, train_label = match_data(train_imgs_mnist, train_labs_mnist, train_data_ch, train_label_ch)
test_data_cochleagrams, test_data_visual, test_label = match_data(test_imgs_mnist, test_labs_mnist, test_data_ch, test_label_ch)

print(len(train_data_cochleagrams))
print(len(train_data_visual))
print(len(train_label))
print(train_label)
print(train_data_cochleagrams)
print(train_data_cochleagrams[0])

print(len(test_data_cochleagrams))
print(len(test_data_visual))
print(len(test_label))

#create random index order

index_visual=np.arange(60000)
print(index_visual)
np.random.shuffle(index_visual)
print(index_visual)

index_auditory=np.arange(60000)
print(index_auditory)
np.random.shuffle(index_auditory)
print(index_auditory)

index_visual_test = np.arange(10000)
np.random.shuffle(index_visual_test)
index_auditory_test = np.arange(10000)
np.random.shuffle(index_auditory_test)
#randomize the order of the cochleagrams and mnist while keeping them coupled to the right label

train_cochleagrams = train_data_cochleagrams[index_auditory]
train_mnist= train_data_visual[index_visual]
train_labels_cochleagrams = train_label[index_auditory]
train_labels_mnist = train_label[index_visual]
print(train_labels_cochleagrams)
print(train_labels_mnist)
print(train_mnist)

test_cochleagrams = test_data_cochleagrams[index_auditory_test]
test_mnist = test_data_visual[index_visual_test]
test_labels_cochleagrams = test_label[index_auditory_test]
test_labels_mnist = test_label[index_visual_test]


# In[60]:


### Task classifier 2 - 4 possible inputs of which 2 are used, addition
#Break training data into blocks
vis1 = train_mnist[:20000]
vis2 = train_mnist[20000:40000]
vis3 = train_mnist[40000:]
vis1_lab = train_labels_mnist[:20000] 
vis2_lab = train_labels_mnist[20000:40000]
vis3_lab = train_labels_mnist[40000:]

aud1 = train_cochleagrams[:20000]
aud2 = train_cochleagrams[20000:40000]
aud3 = train_cochleagrams[40000:]
aud1_lab = train_labels_cochleagrams[:20000]
aud2_lab = train_labels_cochleagrams[20000:40000]
aud3_lab = train_labels_cochleagrams[40000:]

zeros = np.zeros((20000,), dtype=int)
empty_mnist = np.zeros((28,28),dtype=int)

zeros_mnist = np.repeat([empty_mnist], 20000, axis=0)

empty_coch = np.zeros((15,53),dtype=int)
zeros_coch = np.repeat([empty_coch], 20000, axis =0)

#create 4 input streams

in_vis_1 = np.concatenate((vis1, vis2, vis3, zeros_mnist, zeros_mnist, zeros_mnist))
labels_in_vis_1 = np.concatenate((vis1_lab, vis2_lab, vis3_lab, zeros, zeros, zeros))

in_vis_2 = np.concatenate((vis2, zeros_mnist, zeros_mnist, vis3, vis1, zeros_mnist))
labels_in_vis_2 = np.concatenate((vis2_lab, zeros, zeros, vis3_lab, vis1_lab, zeros))

in_aud_1 = np.concatenate((zeros_coch, aud1, zeros_coch, aud2, zeros_coch, aud3))
labels_in_aud_1 = np.concatenate((zeros, aud1_lab, zeros, aud2_lab, zeros, aud3_lab))

in_aud_2 = np.concatenate((zeros_coch, zeros_coch, aud1, zeros_coch, aud3, aud2))
labels_in_aud_2 = np.concatenate((zeros, zeros, aud1_lab, zeros, aud3_lab, aud2_lab))

#Repeat with test data
vis1_test = test_mnist[:3333]
vis2_test = test_mnist[3333:6666]
vis3_test = test_mnist[6666:9999]
vis1_lab_test = test_labels_mnist[:3333]
vis2_lab_test = test_labels_mnist[3333:6666]
vis3_lab_test = test_labels_mnist[6666:9999]

aud1_test = test_cochleagrams[:3333]
aud2_test = test_cochleagrams[3333:6666]
aud3_test = test_cochleagrams[6666:9999]
aud1_lab_test = test_labels_cochleagrams[:3333]
aud2_lab_test = test_labels_cochleagrams[3333:6666]
aud3_lab_test = test_labels_cochleagrams[6666:9999]

zeros_test = np.zeros((3333,), dtype=int)
zeros_mnist_test = np.repeat([empty_mnist], 3333, axis=0)
zeros_coch_test = np.repeat([empty_coch], 3333, axis =0)


in_vis_1_test = np.concatenate((vis1_test, vis2_test, vis3_test, zeros_mnist_test, zeros_mnist_test, zeros_mnist_test))
labels_in_vis_1_test = np.concatenate((vis1_lab_test, vis2_lab_test, vis3_lab_test, zeros_test, zeros_test, zeros_test))

in_vis_2_test = np.concatenate((vis2_test, zeros_mnist_test, zeros_mnist_test, vis3_test, vis1_test, zeros_mnist_test))
labels_in_vis_2_test = np.concatenate((vis2_lab_test, zeros_test, zeros_test, vis3_lab_test, vis1_lab_test, zeros_test))

in_aud_1_test = np.concatenate((zeros_coch_test, aud1_test, zeros_coch_test, aud2_test, zeros_coch_test, aud3_test))
labels_in_aud_1_test = np.concatenate((zeros_test, aud1_lab_test, zeros_test, aud2_lab_test, zeros_test, aud3_lab_test))

in_aud_2_test = np.concatenate((zeros_coch_test, zeros_coch_test, aud1_test, zeros_coch_test, aud3_test, aud2_test))
labels_in_aud_2_test = np.concatenate((zeros_test, zeros_test, aud1_lab_test, zeros_test, aud3_lab_test, aud2_lab_test))



#Addition ....
train_labs_addition_visual_cochleagram = labels_in_vis_1 + labels_in_vis_2 + labels_in_aud_1 + labels_in_aud_2
print(train_labs_addition_visual_cochleagram)
print(max(train_labs_addition_visual_cochleagram))
print(min(train_labs_addition_visual_cochleagram))

test_labs_addition_visual_cochleagram = labels_in_vis_1_test + labels_in_vis_2_test + labels_in_aud_1_test + labels_in_aud_2_test
print(test_labs_addition_visual_cochleagram)
print(max(test_labs_addition_visual_cochleagram))
print(min(test_labs_addition_visual_cochleagram))


# In[ ]:


#Model

#Input Visual 1
input_visual_1 = keras.layers.Input(shape=(28,28))
flatten_vis_1 = keras.layers.Flatten(input_shape=(28,28))(input_visual_1)
l1_vis_1 = keras.layers.Dense(32, activation='sigmoid')(flatten_vis_1)
l2_vis_1 = keras.layers.Dense(10, activation='sigmoid')(l1_vis_1)

#Input Visual 2
input_visual_2 = keras.layers.Input(shape=(28,28))
flatten_vis_2 = keras.layers.Flatten(input_shape=(28,28))(input_visual_2)
l1_vis_2 = keras.layers.Dense(32, activation='sigmoid')(flatten_vis_2)
l2_vis_2 = keras.layers.Dense(10, activation='sigmoid')(l1_vis_2)

#Input auditory 1
input_auditory_1 = keras.layers.Input(shape=(15,53))
flatten_aud_1 = keras.layers.Flatten(input_shape=(15,53))(input_auditory_1)
l1_aud_1 = keras.layers.Dense(32, activation='sigmoid')(flatten_aud_1)
l2_aud_1 = keras.layers.Dense(10, activation='sigmoid')(l1_aud_1)

#Input auditory 2
input_auditory_2 = keras.layers.Input(shape=(15,53))
flatten_aud_2 = keras.layers.Flatten(input_shape=(15,53))(input_auditory_2)
l1_aud_2 = keras.layers.Dense(32, activation='sigmoid')(flatten_aud_2)
l2_aud_2 = keras.layers.Dense(10, activation='sigmoid')(l1_aud_2)


#Concatenated
concatenated = keras.layers.Concatenate()([l2_vis_1, l2_vis_2, l2_aud_1, l2_aud_2])
l3 = keras.layers.Dense(32, activation = 'sigmoid')(concatenated)
out= keras.layers.Dense(19)(l3)

#Bimodal model
model_bimodal = keras.models.Model(inputs=[input_visual_1, input_visual_2, input_auditory_1, input_auditory_2], outputs = out)
model_bimodal.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
params_model_bimodal = model_bimodal.fit([in_vis_1, in_vis_2, in_aud_1, in_aud_2], train_labs_addition_visual_cochleagram, epochs = 40, shuffle=True)

#Evaluate
test_loss, test_acc = model_bimodal.evaluate([in_vis_1_test, in_vis_2_test, in_aud_1_test, in_aud_2_test],  test_labs_addition_visual_cochleagram, verbose=2)
print('test acc:', test_acc)

all_loss_task = params_model_add.history['loss']
all_acc_task = params_model_add.history['acc']
plt.plot(all_loss_task, label='loss')
plt.plot(all_acc_task, label='accuracy')
plt.legend()
plt.show()


# In[ ]:




