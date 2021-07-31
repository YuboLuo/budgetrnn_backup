# this notebook was developed by using tensorflow 2.5.0
import tensorflow as tf
print('tensorflow version:',tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# expand dimension  # from [28*28] to [28*28*1]
x_train = tf.expand_dims(x_train,-1)
x_test = tf.expand_dims(x_test,-1)
y_train = tf.expand_dims(y_train,-1)
y_test = tf.expand_dims(y_test,-1)

print(x_train.shape)

class_names = ['0', '1', '2', '3', '4',
              '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(tf.squeeze(x_train[i]) , cmap=plt.cm.binary) # for tf v2
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

### original test setting
# model = models.Sequential()
# model.add(layers.Conv2D(4, (9, 9), strides = (2,2), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(10))
# model.summary()


model = models.Sequential()
model.add(layers.Conv2D(4, (9, 9), strides = (2,2), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(.2))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10))
model.summary()


# quit()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))