import os

from image_preprocessing import segment_image

dataset_loc = 'dataset/'


def generate_subsquares(img_path):

    img, segments = segment_image(img_path, 200)


if __name__ == '__main__':

    # Iterating through images in dataset
    for image in os.listdir(dataset_loc):

        # Ignore non image files
        if not image.endswith('.jpg'):
            continue

        print('Training on {}....'.format(image))

        subsquares, U, V = generate_subsquares(image)

