import sys

import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.data import imread

if __name__ == '__main__':

    try:
        image_path = str(sys.argv[1])
    except IndexError:
        image_path = input('Image: ')

    image_name = image_path.replace('.jpg', '')
    output_file = 'output/' + image_name + '_SLIC_segments.jpg'

    img = img_as_float(imread(image_path))
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)

    fig = plt.figure("SLIC Super Pixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img, segments_slic))
    plt.axis("off")
    plt.savefig(output_file)


