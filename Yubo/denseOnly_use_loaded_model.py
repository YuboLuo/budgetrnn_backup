import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import initializers
from tensorflow import keras

import numpy as np
from typing import List, Union
from functools import partial
from fxpmath import Fxp
from functools import partial

d = np.load('data.npy',allow_pickle=True)
d = d.item()
train_images = d['TRAIN'][0]
train_labels = d['TRAIN'][1]
test_images = d['TEST'][0]
test_labels = d['TEST'][1]
valid_images = d['VALID'][0]
valid_labels = d['VALID'][1]

model = models.load_model('checkpoint2')

Weights = model.get_weights()

def LeakyReLU(x: np.ndarray):

    return np.where(x > 0, x, x*0.01)


def RELU(x: np.ndarray):

    if not isinstance(x, np.ndarray):
        print(f' RELU only accepts numpy array ')

    return np.multiply(x,np.greater(x,0))



def predict(index: int, images ):
    input = images[index]
    for i in range(int(len(Weights)/2)):

        w,b = Weights[i*2], Weights[i*2+1]
        input = np.add(np.matmul(input,w), b)
        input = LeakyReLU(input)

    return np.argmax(input)

def test(index: int, images, labels ):

    label_pred = predict(index, images)

    if label_pred != labels[index]:
        return False
    else:
        return True







count = 0
for i in range(len(valid_images)):
    if not test(i, valid_images, valid_labels):
        count += 1
print(1 - count/len(valid_images))


