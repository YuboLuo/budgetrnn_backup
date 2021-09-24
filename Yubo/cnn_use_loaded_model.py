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
from keras.models import Model

precision = 10 # fix-point number precision

tofxp = float_to_fp(signed = True, n_bits = 16, n_frac = precision)
tofl = fp_to_float(n_frac = precision)
fxp = partial(Fxp, signed=True, n_word=16, n_frac=precision) # convert float to fix-point


# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# expand dimension
x_train = tf.expand_dims(x_train,-1)
x_test = tf.expand_dims(x_test,-1)
y_train = tf.expand_dims(y_train,-1)
y_test = tf.expand_dims(y_test,-1)

print(x_train.shape)

model = models.load_model('cnn_checkpoint3')  # cnn_checkpoint3, cnn_dropoutDebug
weights = model.get_weights()

### we can use Model to extract the output of each layer and build a new model for each layer
### model.predict() only accepts a batch of inputs, not accepting one single input instance
model_layer0 = Model(inputs = model.input, outputs = model.layers[0].output)
model_layer1 = Model(inputs = model.input, outputs = model.layers[1].output)
model_layer2 = Model(inputs = model.input, outputs = model.layers[2].output)
model_layer3 = Model(inputs = model.input, outputs = model.layers[3].output)


def LeakyReLU(x: np.ndarray):
    '''
    this function works for both float or fxp-fixed-point
    '''
    return np.where(x > 0, x, x*0.01)

def ReLU(x: np.ndarray):
    '''
    this function works for both float or fxp-fixed-point
    '''
    return np.where(x > 0, x, 0)


def conv_calcu_filter_level(mat, filter: Fxp, stride):
    '''
    process one filter
    :param mat: the matrix to be processed
    :param filter: filter matrix
    :param stride:
    :return:
    '''

    assert len(mat.shape) == 2 and len(filter.shape) == 2, 'only process 2d matrix'
    assert mat.shape[0] == mat.shape[1], 'only process square matrix'
    assert filter.shape[0] == filter.shape[1], 'only process square matrix'

    mat_numRows = mat_numCols = mat.shape[0]  # we assume mat is a square-shape matrix
    filter_numRows = filter_numCols = filter.shape[0]  # we assume filter is a square-shape kernel

    result = [] # use a list to store the result

    result_shape = 0
    for i in range(0, mat_numRows, stride):
        if i + filter_numRows > mat_numRows: # when filter moves outside of our matrix
            break

        for j in range(0, mat_numCols, stride):
            if j + filter_numCols > mat_numCols: # when filter moves outside of our matrix
                break

            # (i,j) is the coordinate of the top-left element of the moving filter
            sum = 0
            for m in range(i, i + filter_numRows, 1):
                for n in range(j, j + filter_numCols, 1):
                    sum += mat[m, n] * filter[m - i, n - j]

            result.append(sum)  # we move the filter from left to right and then top to bottom

        result_shape += 1

    result = np.array(result)
    result = result.reshape((result_shape, result_shape))

    return result  # return fix-point number



def conv_calcu_layer_level(input, weights: List, bias, stride):
    '''
    process one entire conv layer
    :param input: the input image
    :param weights: weights for one entire cnn layer
    :param bias: bias for each filter in this layer
    :param stride:
    :return:
    '''
    if len(weights.shape) == 4: # e.g. MNIST: [9,9,1,4]
        weights = weights[:, :, -1, :] # [9,9,1,4] --> [9,9,4]

    num_filter = weights.shape[2]

    size = int((input.shape[0] - weights.shape[0])/stride) + 1 # the matrix size of the result matrix
    result = np.zeros((size, size, num_filter)) # e.g. MNIST: size=10=(28-9)/2+1 for kernel=9, stride=2
    for i in range(weights.shape[2]):

        # process filter by filter
        temp = conv_calcu_filter_level(input, weights[:,:,i], stride=stride)
        result[:,:,i] = temp + bias[i]
        # print(f'finished {i}/{weights.shape[2]}')

    return result

def maxpooling(mat, poolsize):
    '''
    :param mat: the matrix to be processed
    :param poolsize: dimensnion of the pool, assume the pool is square-like
    '''
    assert poolsize > 1, 'poolsize must be larger than 1'

    result_size = int(mat.shape[0] / poolsize)
    num_layer = mat.shape[2]

    result = np.zeros((result_size, result_size, num_layer))
    for layer_idx in range(num_layer):
        for i in range(result_size):
            for j in range(result_size): # get coordinate for result

                # (i, j) is the coordinate of each element after maxpooling
                # (x, y) is the coordinate of top-left element among all corresponding points in the original input matrix

                x, y = i * poolsize, j * poolsize # convert it to coordinate for the input matrix


                max = 0  # initialize max
                for kx in range(poolsize):
                    for ky in range(poolsize):

                        # traverse the entire sub-block that are related to this pooling

                        if mat[x+kx,y+ky,layer_idx] > max:
                            max = mat[x+kx,y+ky,layer_idx]


                # now we get max and save it to result matrix
                result[i,j,layer_idx] = max

    return result



def test(idx,x):
    '''
    test the idx instance of x with our manual implementation of cnn
    '''
    input = np.squeeze(x[idx])
    layer1 = conv_calcu_layer_level(input, weights[0], weights[1], stride = 2)
    pool = maxpooling(ReLU(layer1), 2)
    flat = pool.flatten()  # pay attention how the array is flattened
    dense = ReLU(np.matmul(flat,weights[2]) + weights[3])
    dense1 = np.matmul(dense,weights[4]) + weights[5]
    return np.argmax(dense1)



# define two helper functions, copied from BedgetRNN project
def array_to_string(array: Union[List[Any], np.ndarray]) -> str:
    """
    Formats the 1d array as a comma-separated string enclosed in braces.
    Copied from original BudgetRNN project
    """
    # Validate shapes
    if isinstance(array, np.ndarray):
        assert len(array.shape) == 1, 'Can only format 1d arrays'

    return '{{ {0} }}'.format(','.join(map(str, array)))

def create_matrix(mat: np.ndarray) -> str:
    """
    Converts weight matrix into string
    Copied from original BudgetRNN project
    """
    assert len(mat.shape) == 1 or len(mat.shape) == 2, 'Can only create matrices of at most 2 dimensions'
    # assert mat.shape[0] % 2 == 0 or mat.shape[0] == 1, 'The number of rows must be even or larger than 1. Got: {0}'.format(mat.shape)

    # Ensure the matrix is a 2d array and unpack the dimensions
    if len(mat.shape) == 1:
        mat = np.expand_dims(mat, axis=-1)  # [D, 1]

    matrix_string = array_to_string(mat.reshape(-1))

    return matrix_string

def print_weights_fxp(weights):
    '''
    convert model weights into fix-point arrays
    which you can directly copy to MSP
    :param input: model weights
    '''
    for w in weights:

        if len(w.shape) <= 2:
            w = w.T

            fixed_point = Fxp(w, signed=True, n_word=16, n_frac=10).val
            matrix_string = create_matrix(fixed_point)
            print(matrix_string)

        if len(w.shape) == 4: ### convd layer
            w = w[:,:,0,:]

            for i in range(w.shape[-1]):
                w_filter = w[:,:,i]   # for convd's filter weights, we do not transpose
                fixed_point = Fxp(w_filter, signed=True, n_word=16, n_frac=10).val
                matrix_string = create_matrix(fixed_point)
                print(matrix_string)


def print_input_fxp(input):
    '''
    convert the input image into fix-point array
    which you can directly copy to MSP
    :param input: image instance
    '''
    input = input[:,:,0]   # decrease dimension
    input = input.numpy() # convert tensor to numpy

    fixed_point = Fxp(input, signed=True, n_word=16, n_frac=10).val
    matrix_string = create_matrix(fixed_point)
    print(matrix_string)



### print weights
print_weights_fxp(weights)


### for msp testing
idx = 0
print('\n test input image:')
print_input_fxp(x_train[idx])  # input image
print('label: ',int(y_train[idx])) # label




###### the below is for debug

# count = 0
# for i in range(len(x_test)):
#     if test(i) != int(y_test[i]):
#         count += 1
#
#     if i % 100 == 0:
#         print(f'{i}/{len(x_test)}', 1 - count/(i+0.0000001))
# print(1 - count/len(x_test))
#
#
#
# ### test our manul implementation
# for i in range(len(model.layers)):
#     print(i, model.layers[i]._name)
#
#
#
# ### use the output of an intermediate layer
# layer_output = model.get_layer(model.layers[2]._name).output
# layer_model = models.Model(inputs = model.input, outputs = layer_output)
# pred = layer_model.predict(tf.expand_dims(x_train[0],0))
# # pred = pred[-1,:,:,:]
# print(pred.shape)
#
# ### for verifying maxpool implementation
# a = np.arange(36).reshape((6,6))
# b = np.arange(36).reshape((6,6))
# c = np.stack((a,b), axis=2)
# d = maxpooling(c, 2)





