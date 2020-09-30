#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print(tf.__version__)

mnist =  keras.datasets.mnist
(train_imgs, train_labs), (test_imgs, test_labs) = mnist.load_data()

train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0


# In[ ]:


### Try something with multiple inputs

train_imgs_1=train_imgs[:20000]
train_imgs_2=train_imgs[20000:40000]
train_imgs_3=train_imgs[40000:]
train_labs_1=train_labs[:20000]
train_labs_2=train_labs[20000:40000]
train_labs_3=train_labs[40000:]

train_combo_1=np.concatenate((train_imgs_1,train_imgs_2,train_imgs_3))
train_combo_2=np.concatenate((train_imgs_2,train_imgs_3,train_imgs_1))


train_labs_combo_1=np.concatenate((train_labs_1,train_labs_2,train_labs_3))
train_labs_combo_2=np.concatenate((train_labs_2,train_labs_3,train_labs_1))
train_labs_addition = train_labs_combo_1+train_labs_combo_2

test_imgs_1=test_imgs[:5000]
test_imgs_2=test_imgs[5000:]
test_labs_1=test_labs[:5000]
test_labs_2=test_labs[5000:]
test_labs_addition = test_labs_1+test_labs_2

input1 = keras.layers.Input(shape=(28,28))
flatten1 = keras.layers.Flatten(input_shape=(28,28))(input1)
l1_1 = keras.layers.Dense(32, activation='sigmoid')(flatten1)
l2_1 = keras.layers.Dense(32, activation='sigmoid')(l1_1)
l3_1 = keras.layers.Dense(10, activation='sigmoid')(l2_1)


input2= keras.layers.Input(shape=(28,28))
flatten2 = keras.layers.Flatten(input_shape=(28,28))(input2)
l1_2 = keras.layers.Dense(32, activation='sigmoid')(flatten2)
l2_2 = keras.layers.Dense(32, activation='sigmoid')(l1_2)
l3_2 = keras.layers.Dense(10, activation='sigmoid')(l2_2)

# Use add here instead of concatenated ????

concatenated = keras.layers.Concatenate()([l3_1, l3_2])
l4 = keras.layers.Dense(32, activation = 'sigmoid')(concatenated)
out= keras.layers.Dense(19)(l4)

#def loss_2(y_actual,y_pred):
 #task_loss = kb.square(y_actual-y_pred)
  #  return task_loss

model_2 = keras.models.Model(inputs=[input1, input2], outputs = out)
model_2.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
params_model_add = model_2.fit([train_combo_1, train_combo_2], train_labs_addition, epochs = 25, shuffle=True)

test_loss, test_acc = model_2.evaluate([test_imgs_1, test_imgs_2],  test_labs_addition, verbose=2)
print('test acc:', test_acc)

all_loss_task = params_model_add.history['loss']
all_acc_task = params_model_add.history['acc']
plt.plot(all_loss_task, label='loss')
plt.plot(all_acc_task, label='accuracy')
plt.legend()
plt.show()


# In[53]:


#EXPERIMENTAL: Use * instead of addition

train_imgs_1=train_imgs[:30000]
train_imgs_2=train_imgs[30000:]
train_labs_1=train_labs[:30000]
train_labs_2=train_labs[30000:]
train_labs_times = train_labs_1*train_labs_2

input1 = keras.layers.Input(shape=(28,28))
flatten1 = keras.layers.Flatten(input_shape=(28,28))(input1)
l1 = keras.layers.Dense(16, activation='sigmoid')(flatten1)

input2= keras.layers.Input(shape=(28,28))
flatten2 = keras.layers.Flatten(input_shape=(28,28))(input2)
l2 = keras.layers.Dense(16, activation='sigmoid')(flatten2)

concatenated = keras.layers.Concatenate()([l1, l2])

l3 = keras.layers.Dense(16, activation = 'sigmoid')(concatenated)
out= keras.layers.Dense(10)(l3)

def loss_2(y_actual,y_pred):
    task_loss = kb.square(y_actual-y_pred)
    return task_loss

model_2 = keras.models.Model(inputs=[input1, input2], outputs = out)
model_2.compile(optimizer='adam', loss=loss_2, metrics = ['accuracy'])
params_model_times = model_2.fit([train_imgs_1, train_imgs_2], train_labs_times, epochs = 50, shuffle=True)

test_loss, test_acc = model_task.evaluate(test_imgs,  test_labs, verbose=2)
print('test acc:', test_acc)

all_loss_task = params_model_times.history['loss']
all_acc_task = params_model_times.history['acc']
plt.plot(all_loss_task, label='loss')
plt.plot(all_acc_task, label='accuracy')
plt.legend()
plt.show()


# In[ ]:




