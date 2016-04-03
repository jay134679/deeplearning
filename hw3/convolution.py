#!/usr/bin/python

# Deep Learning Homewor3
# Question 5b
# Alex Pine (akp258)

import numpy as np

# args are two 2d numpy arrays. kernel must be smaller than input.
def convolution(input, kernel):
    result = np.zeros((np.shape(input)[0] - np.shape(kernel)[0] + 1,
                       np.shape(input)[1] - np.shape(kernel)[1] + 1))
    for i in range(np.shape(result)[0]):
        for j in range(np.shape(result)[1]):
            input_window = input[i:i+np.shape(kernel)[0], j:j+np.shape(kernel)[1]]
            result[i][j] = np.sum(input_window * kernel)
    return result

input = np.array([[4,5,2,2,1],[3,3,2,2,4],[4,3,4,1,1],[5,1,4,1,2],[5,1,3,1,4]])
print 'input:', input
kernel = np.array([[4,3,3],[5,5,5],[2,4,3]])
print 'kernel:', kernel
print 'convolution:', convolution(input, kernel)
