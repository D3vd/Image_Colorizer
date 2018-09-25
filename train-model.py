import os
import numpy as np

from image_preprocessing import segment_image
from constants import *

dataset_loc = 'dataset/'


def get_yuv_values(img):
    # Function to get the YUV values from RGB values
    conversion_matrix = np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]).T
    return np.dot(img, conversion_matrix)


def generate_subsquares(img_path):

    img, segments = segment_image(img_path, 200)
    yuv_values = get_yuv_values(img)
    n_segments = segments.max() + 1

    # Compute the Centroids

    # Create array templates
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    U = np.zeros(n_segments)
    V = np.zeros(n_segments)

    # Calculate Centroids
    for (i, j), val in np.ndenumerate(segments):
        point_count[val] += 1
        centroids[val][0] += i
        centroids[val][1] += j
        U[val] += yuv_values[i][j][1]
        V[val] += yuv_values[i][j][2]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        U[k] /= point_count[k]
        V[k] /= point_count[k]


if __name__ == '__main__':

    # Iterating through images in dataset
    # for image in os.listdir(dataset_loc):
    #
    #     # Ignore non image files
    #     if not image.endswith('.jpg'):
    #         continue
    #
    #     print('Training on {}....'.format(image))
    #
    #     subsquares, U, V = generate_subsquares(image)

    file = dataset_loc + '29553856137.jpg'
    print(generate_subsquares(file))

