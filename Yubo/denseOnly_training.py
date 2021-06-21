import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import initializers
from tensorflow import keras

import numpy as np

model = models.Sequential()
initializer = tf.keras.initializers.GlorotUniform()

model.add(Input(shape=(16,)))
model.add(layers.Dense(32, kernel_initializer=initializer))
model.add(layers.LeakyReLU())
model.add(layers.Dense(10, kernel_initializer=initializer))
model.add(layers.LeakyReLU())
model.summary()


d = np.load('data.npy',allow_pickle=True)
d = d.item()
train_images = d['TRAIN'][0]
train_labels = d['TRAIN'][1]
test_images = d['TEST'][0]
test_labels = d['TEST'][1]
valid_images = d['VALID'][0]
valid_labels = d['VALID'][1]

opt = keras.optimizers.Adam(learning_rate=0.01)
# model = models.load_model('checkpoint2')
model.compile(optimizer = opt,
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model.fit(train_images, train_labels, batch_size = 32, verbose = 2,  epochs=10, validation_data=(test_images, test_labels))


