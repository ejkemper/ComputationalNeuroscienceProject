import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print(tf.__version__)

mnist =  keras.datasets.mnist
(train_imgs, train_labs), (test_imgs, test_labs) = mnist.load_data()

#plt.figure()
#plt.imshow(train_imgs[0])
#plt.show()

train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(16, activation='sigmoid'),
    keras.layers.Dense(16, activation='sigmoid'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

params = model.fit(train_imgs, train_labs, epochs=20, shuffle=True)
test_loss, test_acc = model.evaluate(test_imgs,  test_labs, verbose=2)
print('test acc:', test_acc)

all_loss = params.history['loss']
all_acc = params.history['accuracy']
plt.plot(all_loss, label='loss')
plt.plot(all_acc, label='accuracy')
plt.legend()
plt.show()
