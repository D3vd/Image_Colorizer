import numpy as np

SQUARE_SIZE = 10

C = .125
EPSILON = .0625

RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -.58060],
                         [1, 2.03211, 0]]).T
YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T

U_MAX = 0.436
V_MAX = 0.615

ICM_ITERATIONS = 10
COVAR = 0.25
WEIGHT_DIFF = 2
ITER_EPSILON = .01
THRESHOLD = 25
