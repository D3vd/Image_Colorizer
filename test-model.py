import sys

from sklearn.externals import joblib
from skimage.io import imsave

import matplotlib.pyplot as plt

from constants import *
from image_preprocessing import segment_image


def clamp(val, low, high):
    return np.maximum(np.minimum(val, high), low)


def clamp_u(val):
    return clamp(val, -U_MAX, U_MAX)


def clamp_v(val):
    return clamp(val, -V_MAX, U_MAX)


def get_rgb_values(img):
    return clamp(np.dot(img, RGB_FROM_YUV), 0, 1)


def get_yuv_values(img):
    return np.dot(img, YUV_FROM_RGB)


def generate_adjacencies(segments, n_segments, img, subsquares):
    adjacency_list = []
    for i in range(n_segments):
        adjacency_list.append(set())
    for (i, j), value in np.ndenumerate(segments):
        if i < img.shape[0] - 1:
            new_value = segments[i + 1][j]
            if value != new_value and np.linalg.norm(subsquares[value] - subsquares[new_value]) < THRESHOLD:
                adjacency_list[value].add(new_value)
                adjacency_list[new_value].add(value)

        if j < img.shape[1] - 1:
            new_value = segments[i][j + 1]
            if value != new_value and np.linalg.norm(subsquares[value] - subsquares[new_value]) < THRESHOLD:
                adjacency_list[value].add(new_value)
                adjacency_list[new_value].add(value)

    return adjacency_list


def apply_mrf(observed_u, observed_v, segments, n_segments, img, subsquares):
    hidden_u = np.copy(observed_u)
    hidden_v = np.copy(observed_v)

    adjacency_list = generate_adjacencies(segments, n_segments, img, subsquares)

    for iteration in range(ICM_ITERATIONS):
        new_u = np.zeros(n_segments)
        new_v = np.zeros(n_segments)

        for k in range(n_segments):

            u_potential = 100000
            v_potential = 100000
            u_min = -1
            v_min = -1

            for u in np.arange(-U_MAX, U_MAX, .001):
                u_computed = (u - observed_u[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    u_computed += WEIGHT_DIFF * ((u - hidden_u[adjacency]) ** 2)
                if u_computed < u_potential:
                    u_potential = u_computed
                    u_min = u
            new_u[k] = u_min

            for v in np.arange(-V_MAX, V_MAX, .001):
                v_computed = (v - observed_v[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    v_computed += WEIGHT_DIFF * ((v - hidden_v[adjacency]) ** 2)
                if v_computed < v_potential:
                    v_potential = v_computed
                    v_min = v
            new_v[k] = v_min

        u_diff = np.linalg.norm(hidden_u - new_u)
        v_diff = np.linalg.norm(hidden_v - new_v)
        hidden_u = new_u
        hidden_v = new_v
        if u_diff < ITER_EPSILON and v_diff < ITER_EPSILON:
            break

    return hidden_u, hidden_v


def predict_image(u_svr, v_svr, path, verbose, output_file=None):
    img, segments = segment_image(path, 200)
    yuv = get_yuv_values(img)
    n_segments = segments.max() + 1

    # Create Centroids Template
    point_count = np.zeros(n_segments)
    centroids = np.zeros((n_segments, 2))
    luminance = np.zeros(n_segments)

    for (i, j), value in np.ndenumerate(segments):
        point_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j
        luminance[value] += yuv[i][j][0]

    for k in range(n_segments):
        centroids[k] /= point_count[k]
        luminance[k] /= point_count[k]

    # Generate Subsquares
    subsquares = np.zeros((n_segments, SQUARE_SIZE * SQUARE_SIZE))
    for k in range(n_segments):
        top = max(int(centroids[k][0]), 0)
        if top + SQUARE_SIZE >= img.shape[0]:
            top = img.shape[0] - 1 - SQUARE_SIZE
        left = max(int(centroids[k][1]), 0)
        if left + SQUARE_SIZE >= img.shape[1]:
            left = img.shape[1] - 1 - SQUARE_SIZE
        for i in range(0, SQUARE_SIZE):
            for j in range(0, SQUARE_SIZE):
                subsquares[k][i * SQUARE_SIZE + j] = yuv[i + top][j + left][0]
        subsquares[k] = np.fft.fft2(subsquares[k].reshape(SQUARE_SIZE, SQUARE_SIZE)).reshape(SQUARE_SIZE * SQUARE_SIZE)

    # Predict using SVR
    predicted_u = clamp_u(u_svr.predict(subsquares) * 2)
    predicted_v = clamp_v(v_svr.predict(subsquares) * 2)

    # Apply MRF to smooth out coloring
    predicted_u, predicted_v = apply_mrf(predicted_u, predicted_v, segments, n_segments, img, subsquares)

    # Reconstruct the image
    for (i, j), value in np.ndenumerate(segments):
        yuv[i][j][1] = predicted_u[value]
        yuv[i][j][2] = predicted_v[value]
    rgb = get_rgb_values(yuv)

    if verbose:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(rgb)
        if output_file:
            imsave(output_file, rgb)
        plt.show()


if __name__ == '__main__':
    try:
        image_path = str(sys.argv[1])
    except IndexError:
        image_path = input('Image: ')

    image_name = image_path.replace('.jpg', '')
    output_file = 'output/' + image_name + '_out.jpg'

    # Load trained models
    u_svr = joblib.load('models/u_svr.model')
    v_svr = joblib.load('models/v_svr.model')
    predict_image(u_svr, v_svr, image_path, True, output_file)
