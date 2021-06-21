import tensorflow as tf
print('Tensorflow Version:',tf.__version__)

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras import initializers
from tensorflow import keras

import numpy as np
from typing import List, Union, Any
from functools import partial
from fxpmath import Fxp
from functools import partial
from rig.type_casts import float_to_fp, fp_to_float


# load data
d = np.load('data.npy',allow_pickle=True)
d = d.item()
train_images = d['TRAIN'][0]
train_labels = d['TRAIN'][1]
test_images = d['TEST'][0]
test_labels = d['TEST'][1]
valid_images = d['VALID'][0]
valid_labels = d['VALID'][1]

def LeakyReLU(x: np.ndarray):

    return np.where(x > 0, x, x*0.01)

def fxp_matrix_multiple(mat1: Fxp, mat2: Fxp) -> Fxp:
    """
    for fixed-point matrix mulplication
    """
    m = mat1.shape[0]
    n = mat2.shape[1]

    if len(mat1.shape) == 2: # if mat1 is 2d array
        result = Fxp(np.zeros((m,n)), signed = True, n_word = 16, n_frac = 10)
        for i in range(m):
            for j in range(n):
                for k in range(mat2.shape[0]):
                    result[i][j] += mat1[i][k] * mat2[k][j]

        return result

    else: # if mat1 is 1d array
        result =  Fxp(np.zeros((n)), signed = True, n_word = 16, n_frac = 10)
        for j in range(n):
            for k in range(mat2.shape[0]):
                result[j] += mat1[k] * mat2[k][j]

        return result

def predict(index: int, images ):
    """
    make a prediction on one image instance
    """
    input = images[index]
    for i in range(int(len(Weights)/2)):

        w,b = Weights[i*2], Weights[i*2+1]
        input = np.add(np.matmul(input,w), b)
        input = LeakyReLU(input)

    np.argmax(input)
    return input, np.argmax(input)

def predict_fxp(index: int, images ):
    """
    fixed-point version
    make a prediction on one image instance
    """
    input = Fxp(images[index], signed = True, n_word = 16, n_frac = 10)
    for i in range(int(len(Weights)/2)):

        w = Fxp(Weights[i * 2],     signed = True, n_word = 16, n_frac = 10)
        b = Fxp(Weights[i * 2 + 1], signed=True, n_word=16, n_frac=10)
        input = fxp_matrix_multiple(input, w)
        input = input + b
        input = LeakyReLU(input)

    np.argmax(input)
    return np.argmax(input)

def test_fxp(index: int, images, labels ):

    label_pred = predict_fxp(index, images)

    if label_pred != labels[index]:
        return False
    else:
        return True

def float_to_fixed_point(x: float, precision: int) -> int:
    """
    Converts the given floating point value to fixed point representation
    with the given number of fractional bits.
    """
    multiplier = 1 << precision

    width = 16 if precision >= 8 else 8
    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    fp_val = int(round(x * multiplier))

    if fp_val > max_val:
        print('WARNING: Observed positive overflow')
        return max_val
    elif fp_val < min_val:
        print('WARNING: Observed negative overflow')
        return min_val
    return fp_val


def tensor_to_fixed_point(tensor: Union[List[float], np.ndarray], precision: int) -> Union[List[int], np.ndarray]:
    """
    Converts each element in the given tensor to fixed point representation with the given
    number of fractional bits.
    """
    fixed_point_converter = partial(float_to_fixed_point, precision=precision)

    if isinstance(tensor, np.ndarray):
        fp_function = np.vectorize(fixed_point_converter)
        return fp_function(tensor)
    else:
        return list(map(fixed_point_converter, tensor))


def array_to_string(array: Union[List[Any], np.ndarray]) -> str:
    """
    Formats the 1d array as a comma-separated string enclosed in braces.
    """
    # Validate shapes
    if isinstance(array, np.ndarray):
        assert len(array.shape) == 1, 'Can only format 1d arrays'

    return '{{ {0} }}'.format(','.join(map(str, array)))


def create_matrix(mat: np.ndarray) -> str:
    """
    Converts weight matrix into string
    """
    assert len(mat.shape) == 1 or len(mat.shape) == 2, 'Can only create matrices of at most 2 dimensions'
    assert mat.shape[0] % 2 == 0 or mat.shape[0] == 1, 'The number of rows must be even or larger than 1. Got: {0}'.format(mat.shape)

    # Ensure the matrix is a 2d array and unpack the dimensions
    if len(mat.shape) == 1:
        mat = np.expand_dims(mat, axis=-1)  # [D, 1]

    matrix_string = array_to_string(mat.reshape(-1))

    return matrix_string

# load model
model = models.load_model('checkpoint2')

Weights = model.get_weights()

for w in Weights:

    # For 2d matrices, we always transpose the weights. In Tensorflow, a standard dense layer uses the format
    # (x^T)(W). In the embedded implementation, we instead use (W^T)x. This is purely a convention--the embedded
    # implementation uses a row-based vector format.
    if len(w.shape) == 2:
        w = w.T

    # fixed_point = tensor_to_fixed_point(w,precision=10)
    fixed_point = Fxp(w, signed = True, n_word = 16, n_frac = 10).val
    matrix_string = create_matrix(fixed_point)
    # break
    print(matrix_string)

def get_test_sample_msp(idx: int):


    temp = fxp(valid_images[idx]).val
    temp = np.expand_dims(temp, axis=1)
    temp = (np.concatenate((temp, np.zeros_like(temp)), axis=1)).flatten()

    return temp, valid_labels[idx]

fxp = partial(Fxp, signed = True, n_word = 16, n_frac = 10)
x = fxp(w)




input = fxp(valid_images[0])
w1 = fxp(Weights[0])
b1 = fxp(Weights[1])
w2 = fxp(Weights[2])
b2 = fxp(Weights[3])

output1 = fxp_matrix_multiple(input,w1)
predict_fxp(0,valid_images)

