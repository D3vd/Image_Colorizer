from skimage.util import img_as_float
from skimage.data import imread
from skimage.segmentation import slic


# Segmentation is done to change the representation
# of the image into something that is more
# meaningful and easier to analyze

def segment_image(img_path, n_segments):
    img = img_as_float(imread(img_path))

    print('Segmenting Image {}'.format(img_path))
    # TODO: Save segment files?
    segments = slic(img, n_segments, compactness=10, sigma=1)

    return img, segments
