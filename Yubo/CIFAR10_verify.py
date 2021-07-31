import tensorflow as tf
print('tensorflow version:',tf.__version__)

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models, Input
from tensorflow import keras

# setting class names
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()



