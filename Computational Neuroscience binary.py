#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

train_imgs_mnist = train_imgs_mnist/255.0
test_imgs_mnist = test_imgs_mnist/255.0

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


# In[3]:


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

#create random index order

index_visual=np.arange(60000)
np.random.shuffle(index_visual)

index_auditory=np.arange(60000)
np.random.shuffle(index_auditory)


index_visual_test = np.arange(10000)
np.random.shuffle(index_visual_test)
index_auditory_test = np.arange(10000)
np.random.shuffle(index_auditory_test)
#randomize the order of the cochleagrams and mnist while keeping them coupled to the right label

train_cochleagrams = train_data_cochleagrams[index_auditory]
train_mnist= train_data_visual[index_visual]
train_labels_cochleagrams = train_label[index_auditory]
train_labels_mnist = train_label[index_visual]


test_cochleagrams = test_data_cochleagrams[index_auditory_test]
test_mnist = test_data_visual[index_visual_test]
test_labels_cochleagrams = test_label[index_auditory_test]
test_labels_mnist = test_label[index_visual_test]


# In[4]:


### Prepare data for the task classifiers 
#Break training data into blocks
vis1 = train_mnist[:30000]
vis2 = train_mnist[30000:]
vis1_lab = train_labels_mnist[:30000] 
vis2_lab = train_labels_mnist[30000:]

aud1 = train_cochleagrams[:30000]
aud2 = train_cochleagrams[30000:]
aud1_lab = train_labels_cochleagrams[:30000]
aud2_lab = train_labels_cochleagrams[30000:]

zeros = np.zeros((30000,), dtype=int)
empty_mnist = np.zeros((28,28),dtype=int)
zeros_mnist = np.repeat([empty_mnist], 30000, axis=0)
empty_coch = np.zeros((15,53),dtype=int)
zeros_coch = np.repeat([empty_coch], 30000, axis =0)

#create 4 input streams
## vis1  vis2  0    0 
## vis2   0    0    aud1
## 0     vis1 aud2  0 
## 0      0   aud1  aud2
in_vis_1 = np.concatenate((vis1, vis2, zeros_mnist, zeros_mnist))
labels_in_vis_1 = np.concatenate((vis1_lab, vis2_lab, zeros, zeros))

in_vis_2 = np.concatenate((vis2, zeros_mnist, vis1, zeros_mnist))
labels_in_vis_2 = np.concatenate((vis2_lab, zeros, vis1_lab, zeros))

in_aud_1 = np.concatenate((zeros_coch, zeros_coch, aud2, aud1))
labels_in_aud_1 = np.concatenate((zeros, zeros, aud2_lab, aud1_lab))

in_aud_2 = np.concatenate((zeros_coch, aud1, zeros_coch, aud2))
labels_in_aud_2 = np.concatenate((zeros, aud1_lab, zeros, aud2_lab))

#Repeat with test data
vis1_test = test_mnist[:5000]
vis2_test = test_mnist[5000:]
vis1_lab_test = test_labels_mnist[:5000]
vis2_lab_test = test_labels_mnist[5000:]

aud1_test = test_cochleagrams[:5000]
aud2_test = test_cochleagrams[5000:]
aud1_lab_test = test_labels_cochleagrams[:5000]
aud2_lab_test = test_labels_cochleagrams[5000:]

zeros_test = np.zeros((5000,), dtype=int)
zeros_mnist_test = np.repeat([empty_mnist], 5000, axis=0)
zeros_coch_test = np.repeat([empty_coch], 5000, axis =0)

#create 4 input streams
## vis1  vis2  0    0 
## vis2   0    0    aud1
## 0     vis1 aud2  0 
## 0      0   aud1  aud2

in_vis_1_test = np.concatenate((vis1_test, vis2_test, zeros_mnist_test, zeros_mnist_test))
labels_in_vis_1_test = np.concatenate((vis1_lab_test, vis2_lab_test, zeros_test, zeros_test))

in_vis_2_test = np.concatenate((vis2_test, zeros_mnist_test, vis1_test, zeros_mnist_test))
labels_in_vis_2_test = np.concatenate((vis2_lab_test, zeros_test, vis1_lab_test, zeros_test))

in_aud_1_test = np.concatenate((zeros_coch_test, zeros_coch_test, aud2_test, aud1_test))
labels_in_aud_1_test = np.concatenate((zeros_test, zeros_test, aud2_lab_test, aud1_lab_test))

in_aud_2_test = np.concatenate((zeros_coch_test, aud1_test, zeros_coch_test, aud2_test))
labels_in_aud_2_test = np.concatenate((zeros_test, aud1_lab_test, zeros_test, aud2_lab_test))



# In[5]:


from tensorflow.python.keras import Model
class SequentialmodelBinary(Model):
    def __init__(self):
        super(SequentialmodelBinary, self).__init__()
        #Input Visual  sequential
        self.input_mnist1 = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(32, activation='relu')
        ])
        
        #Input Visual  sequential
        self.input_mnist2 = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(32, activation='relu')
        ])


        #Input auditory sequential
        self.input_coch1 = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15,53)),
            keras.layers.Dense(32, activation='relu')
        ])
        
         #Input auditory sequential
        self.input_coch2 = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=(15,53)),
            keras.layers.Dense(32, activation='relu')
        ])

        #Concatenated sequential 
        self.concatenated = tf.keras.Sequential([
            keras.layers.Concatenate(axis=-1),
            keras.layers.Dense(64, activation = 'relu'),
            keras.layers.Dense(32, activation = 'relu')
        ])
        
        #output sequential
        self.out= tf.keras.Sequential([keras.layers.Dense(1)])
        

    
    def encode(self, im1, im2, ch1, ch2):
        encoded_im1 = self.input_mnist1(im1) 
        encoded_im2 = self.input_mnist2(im2)
        encoded_ch1 = self.input_coch1(ch1)
        encoded_ch2 = self.input_coch2(ch2)
        combined_encoding = self.concatenated([encoded_im1, encoded_im2, encoded_ch1, encoded_ch2])
        return combined_encoding
    
    def output(self, encoded):
        out = self.out(encoded)
        return out
    
    def call(self, input):
        im1, im2, ch1, ch2 = input
        encoding = self.encode(im1, im2, ch1, ch2)
        out = self.output(encoding)
        return out
        
    


# In[ ]:


#OUD Input Visual 1
input_visual_1 = keras.layers.Input(shape=(28,28))
flatten_vis_1 = keras.layers.Flatten(input_shape=(28,28))(input_visual_1)
l1_vis_1 = keras.layers.Dense(32, activation='relu')(flatten_vis_1)

#Input Visual 2
input_visual_2 = keras.layers.Input(shape=(28,28))
flatten_vis_2 = keras.layers.Flatten(input_shape=(28,28))(input_visual_2)
l1_vis_2 = keras.layers.Dense(32, activation='relu')(flatten_vis_2)

#Input auditory 1
input_auditory_1 = keras.layers.Input(shape=(15,53))
flatten_aud_1 = keras.layers.Flatten(input_shape=(15,53))(input_auditory_1)
l1_aud_1 = keras.layers.Dense(32, activation='relu')(flatten_aud_1)

#Input auditory 2
input_auditory_2 = keras.layers.Input(shape=(15,53))
flatten_aud_2 = keras.layers.Flatten(input_shape=(15,53))(input_auditory_2)
l1_aud_2 = keras.layers.Dense(32, activation='relu')(flatten_aud_2)

#Concatenated
concatenated = keras.layers.Concatenate()([l1_vis_1, l1_vis_2, l1_aud_1, l1_aud_2])
l2 = keras.layers.Dense(64, activation = 'relu')(concatenated)
l3 = keras.layers.Dense(32, activation = 'relu')(l2)
out= keras.layers.Dense(1)(l3)


# In[6]:


train_labs_bigger = []
train_labs_bigger[:30000] = labels_in_vis_1[:30000] > labels_in_vis_2[:30000]
train_labs_bigger[30000:60000] = labels_in_vis_1[30000:60000] > labels_in_aud_2[30000:60000]
train_labs_bigger[60000:90000] = labels_in_aud_1[60000:90000] > labels_in_vis_2[60000:90000]
train_labs_bigger[90000:] = labels_in_aud_1[90000:] > labels_in_aud_2[90000:]
train_labs_bigger = np.asarray(train_labs_bigger)
#print(train_labs_bigger)

#test_labs_addition_visual_cochleagram = 
test_labs_bigger = []
test_labs_bigger[:5000] = labels_in_vis_1_test[:5000] > labels_in_vis_2_test[:5000]
test_labs_bigger[5000:10000] = labels_in_vis_1_test[5000:10000] > labels_in_aud_2_test[5000:10000]
test_labs_bigger[10000:15000] = labels_in_aud_1_test[10000:15000] > labels_in_vis_2_test[10000:15000]
test_labs_bigger[15000:] = labels_in_aud_1_test[15000:] > labels_in_aud_2_test[15000:]
test_labs_bigger = np.asarray(test_labs_bigger)


# In[7]:


model_big = SequentialmodelBinary()
model_big.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model_big.fit([in_vis_1, in_vis_2, in_aud_1, in_aud_2], train_labs_bigger, epochs = 5, batch_size=32, shuffle = True)

test_loss, test_acc = model_big.evaluate([in_vis_1_test, in_vis_2_test, in_aud_1_test, in_aud_2_test],  test_labs_bigger, verbose=2)
print('test acc:', test_acc)


# In[8]:


def make_dataset(data_im, data_ch):
    empty_mnist = np.zeros((28,28),dtype=float)
    empty_coch = np.zeros((15,53),dtype=float)

    length = int(len(data_im)/2)
    print(length)
    data_im_zeros = np.repeat([empty_mnist], length, axis=0)
    data_ch_zeros = np.repeat([empty_coch], length, axis=0)
    
    data_im1 = data_im[:length]
    data_im2 = data_im[length:2*length]
    data_ch1 = data_ch[:length]
    data_ch2 = data_ch[length:2*length]
    

    data_1 = np.concatenate((data_im1, data_im2, data_im_zeros, data_im_zeros))
    data_2 = np.concatenate((data_im2, data_im1, data_im_zeros, data_im_zeros))
    data_3 = np.concatenate((data_ch_zeros, data_ch_zeros, data_ch1, data_ch2))
    data_4 = np.concatenate((data_ch_zeros, data_ch_zeros, data_ch2, data_ch1))
    data = [data_1, data_2, data_3, data_4]

    #if input is an image, output 1, if input is a cochleagram output 0
    im_lab = np.ones(2*length)
    ch_lab = np.zeros(2*length)
    labels = np.concatenate([im_lab, ch_lab])

    
    return data, labels


# In[9]:


class Classifier(Model):
    def __init__(self, model):
        super(Classifier, self).__init__()
        # because we have a binary classifier, sigmoid is more suitable
        activation = 'sigmoid'
        # self.model: keras.Model = tf.keras.models.load_model(dir, custom_objects={"seperate_loss": seperate_loss})
        self.model = model
        for layer in self.model.layers:
            layer.trainable = False

        self.classifier = tf.keras.Sequential([
            keras.layers.Dense(1, activation=activation)  # latent space
        ])

    def call(self, input):
        im1, im2, ch1, ch2 = input
        latent_space = model_big.encode(im1, im2, ch1, ch2)
        classified = self.classifier(latent_space)
        return classified


# In[10]:


data_ls, train_labels_latent_space = make_dataset(train_mnist, train_cochleagrams)
data_testls, test_labels_latent_space = make_dataset(test_mnist, test_cochleagrams)
print(len(train_labels_latent_space))

classifier = Classifier(model_big)
classifier.compile(optimizer='adam',
                       loss=keras.losses.BinaryCrossentropy(from_logits=True),
                       # since input are values, no probabilities
                       metrics=['accuracy'])


history = classifier.fit(data_ls, train_labels_latent_space, batch_size=32, steps_per_epoch=None, epochs=5, shuffle=True, validation_data=(data_testls, test_labels_latent_space))
#test_loss, test_acc = classifier.evaluate([data_testls_1, data_testls_2, data_testls_3, data_testls_4], test_labels_latent_space, verbose=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/biggerthanaccuracy')
plt.show()


# In[ ]:


#OLD Bigger than
model_bigger = keras.models.Model(inputs=[input_visual_1, input_visual_2, input_auditory_1, input_auditory_2], outputs = out)
model_bigger.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
params_model_bigger = model_bigger.fit([in_vis_1, in_vis_2, in_aud_1, in_aud_2], train_labs_bigger, epochs = 2, shuffle=True)

#Evaluate model bigger than
test_loss, test_acc = model_bigger.evaluate([in_vis_1_test, in_vis_2_test, in_aud_1_test, in_aud_2_test],  test_labs_bigger, verbose=2)
print('test acc:', test_acc)

all_loss_task = params_model_bigger.history['loss']
all_acc_task = params_model_bigger.history['acc']
plt.plot(all_loss_task, label='loss')
plt.plot(all_acc_task, label='accuracy')
plt.legend()
plt.show()


# In[12]:


labels_digits = np.concatenate([test_labels_mnist, test_labels_cochleagrams])
digits_0 = np.where(labels_digits ==0)[0]
digits_1 = np.where(labels_digits == 1)[0]
digits_2 = np.where(labels_digits == 2)[0]
digits_3 = np.where(labels_digits == 3)[0]
digits_4 = np.where(labels_digits == 4)[0]
digits_5 = np.where(labels_digits == 5)[0]
digits_6 = np.where(labels_digits == 6)[0]
digits_7 = np.where(labels_digits == 7)[0]
digits_8 = np.where(labels_digits == 8)[0]
digits_9 = np.where(labels_digits == 9)[0]

#digit_0_mnist = digits_0[:int(len(digits_0)/2)]
digit_0_mnist = digits_0[:int(len(digits_0)/2)]
digit_1_mnist = digits_1[:int(len(digits_1)/2)]
digit_2_mnist = digits_2[:int(len(digits_2)/2)]
digit_3_mnist = digits_3[:int(len(digits_3)/2)]
digit_4_mnist = digits_4[:int(len(digits_4)/2)]
digit_5_mnist = digits_5[:int(len(digits_5)/2)]
digit_6_mnist = digits_6[:int(len(digits_6)/2)]
digit_7_mnist = digits_7[:int(len(digits_7)/2)]
digit_8_mnist = digits_8[:int(len(digits_8)/2)]
digit_9_mnist = digits_9[:int(len(digits_9)/2)]

digit_0_coch = digits_0[int(len(digits_0)/2):]
digit_1_coch = digits_1[int(len(digits_1)/2):]
digit_2_coch = digits_2[int(len(digits_2)/2):]
digit_3_coch = digits_3[int(len(digits_3)/2):]
digit_4_coch = digits_4[int(len(digits_4)/2):]
digit_5_coch = digits_5[int(len(digits_5)/2):]
digit_6_coch = digits_6[int(len(digits_6)/2):]
digit_7_coch = digits_7[int(len(digits_7)/2):]
digit_8_coch = digits_8[int(len(digits_8)/2):]
digit_9_coch = digits_9[int(len(digits_9)/2):]

print(digit_9_mnist)
print(digit_9_coch)


# In[14]:


from sklearn.manifold import TSNE

latent = model_big.encode(data_testls[0], data_testls[1], data_testls[2], data_testls[3])
X = latent
X_embedded = TSNE(n_components=2, perplexity=8, init='pca').fit_transform(X)
#print(X_embedded.shape)


# In[18]:


plt.scatter(X_embedded[digit_0_mnist, 0], X_embedded[digit_0_mnist, 1],  color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_1_mnist, 0], X_embedded[digit_1_mnist, 1],  color="tab:orange", marker="o", s=1)
plt.scatter(X_embedded[digit_2_mnist, 0], X_embedded[digit_2_mnist, 1],  color="tab:green", marker="o", s=1)
plt.scatter(X_embedded[digit_3_mnist, 0], X_embedded[digit_3_mnist, 1], color="tab:red", marker="o", s=1)
plt.scatter(X_embedded[digit_4_mnist, 0], X_embedded[digit_4_mnist, 1], color="tab:purple", marker="o", s=1)
plt.scatter(X_embedded[digit_6_mnist, 0], X_embedded[digit_6_mnist, 1],  color="tab:pink", marker="o", s=1)
plt.scatter(X_embedded[digit_7_mnist, 0], X_embedded[digit_7_mnist, 1], color="tab:gray", marker="o", s=1)
plt.scatter(X_embedded[digit_8_mnist, 0], X_embedded[digit_8_mnist, 1], color="tab:olive", marker="o", s=1)
plt.scatter(X_embedded[digit_9_mnist, 0], X_embedded[digit_9_mnist, 1],  color="tab:cyan", marker="o", s=1)


plt.scatter(X_embedded[digit_0_mnist, 0], X_embedded[digit_0_coch, 1],  color="tab:blue", marker="s", s=2)
plt.scatter(X_embedded[digit_1_mnist, 0], X_embedded[digit_1_coch, 1],  color="tab:orange", marker="s", s=2)
plt.scatter(X_embedded[digit_2_mnist, 0], X_embedded[digit_2_coch, 1],  color="tab:green", marker="s", s=2)
plt.scatter(X_embedded[digit_3_mnist, 0], X_embedded[digit_3_coch, 1],  color="tab:red", marker="s", s=2)
plt.scatter(X_embedded[digit_4_mnist, 0], X_embedded[digit_4_coch, 1],  color="tab:purple", marker="s", s=2)
plt.scatter(X_embedded[digit_5_mnist, 0], X_embedded[digit_5_coch, 1],  color="tab:brown", marker="s", s=2)
plt.scatter(X_embedded[digit_6_mnist, 0], X_embedded[digit_6_coch, 1],  color="tab:pink", marker="s", s=2)
plt.scatter(X_embedded[digit_7_mnist, 0], X_embedded[digit_7_coch, 1],  color="tab:gray", marker="s", s=2)
plt.scatter(X_embedded[digit_8_mnist, 0], X_embedded[digit_8_coch, 1],  color="tab:olive", marker="s", s=2)
plt.scatter(X_embedded[digit_9_mnist, 0], X_embedded[digit_9_coch, 1],  color="tab:cyan", marker="s", s=2)
plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.ylabel("Dim 2")
plt.xlabel("Dim 1")
#plt.legend(["0 mnist","1 mnist", "2 mnist", "3 mnist", "4 mnist", "5 mnist", "6 mnist", "7 mnist", "8 mnist", "9 mnist", "0 coch", "1 coch", "2 coch", "3 coch", "4 coch", "5 coch", "6 coch", "7 coch", "8 coch", "9 coch"])
plt.savefig('C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/biggerthantsnedigits')
plt.show()


# In[19]:


plt.scatter(X_embedded[digit_0_mnist, 0], X_embedded[digit_0_mnist, 1],  color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_0_mnist, 0], X_embedded[digit_0_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_1_mnist, 0], X_embedded[digit_1_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_1_mnist, 0], X_embedded[digit_1_mnist, 1],  color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_2_mnist, 0], X_embedded[digit_2_mnist, 1],  color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_2_mnist, 0], X_embedded[digit_2_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_3_mnist, 0], X_embedded[digit_3_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_3_mnist, 0], X_embedded[digit_3_mnist, 1], color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_4_mnist, 0], X_embedded[digit_4_mnist, 1], color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_4_mnist, 0], X_embedded[digit_4_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_5_mnist, 0], X_embedded[digit_5_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_6_mnist, 0], X_embedded[digit_6_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_7_mnist, 0], X_embedded[digit_7_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_5_mnist, 0], X_embedded[digit_5_mnist, 1],  color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_6_mnist, 0], X_embedded[digit_6_mnist, 1],  color="tab:blue", marker="o", s=1)

plt.scatter(X_embedded[digit_7_mnist, 0], X_embedded[digit_7_mnist, 1], color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_8_mnist, 0], X_embedded[digit_8_mnist, 1], color="tab:blue", marker="o", s=1)
plt.scatter(X_embedded[digit_8_mnist, 0], X_embedded[digit_8_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_9_mnist, 0], X_embedded[digit_9_coch, 1],  color="tab:green", marker="s", s=1)
plt.scatter(X_embedded[digit_9_mnist, 0], X_embedded[digit_9_mnist, 1],  color="tab:blue", marker="o", s=1)






plt.grid(color='k', linestyle='-', linewidth=0.25)
plt.ylabel("Dim 2")
plt.xlabel("Dim 1")
plt.savefig('C:/Users/s141554/Documents/Systems Biology/Computational neuroscience/biggerthantsnecochmnist')
plt.show()

