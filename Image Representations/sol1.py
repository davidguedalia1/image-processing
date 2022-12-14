import numpy as np
import pip as plt
from skimage.color import rgb2gray
from imageio import imread

GRAY = 1
TRANSFER_MATRIX_YIQ_RGB = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def image_float(filename):
    '''

    :param filename:
    :return:
    '''
    image = imread(filename)
    image_to_return = image.astype(np.float64)
    image_to_return /= 255
    return image_to_return


def read_image(filename, representation):
    """
    :param filename: name of file containing image
    :param representation: e.g. 1 for grayscale, 2 for RGB, 333 for default (no change)
    :return: matrix representing the image, with normalized intensities in [0,1]
    :raise: SystemExit if file not found / error in file.
    """
    image = image_float(filename)
    if representation == GRAY:
        return rgb2gray(image)
    return image


def imdisplay(filename, representation):
    """
    displays an image from a given filename in the given representation.
    """
    image = read_image(filename, representation)
    if representation == GRAY:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)


def rgb2yiq_yiq2rgb(im, transfer_matrix):
    '''

    :param im:
    :param transfer_matrix:
    :return:
    '''
    r_y = im[:, :, 0]
    g_i = im[:, :, 1]
    b_q = im[:, :, 2]
    image_yiq_rgb = np.empty(im.shape)
    image_yiq_rgb[:, :, 0] = r_y * transfer_matrix[0][0] + g_i * transfer_matrix[0][1] + b_q * transfer_matrix[0][2]
    image_yiq_rgb[:, :, 1] = r_y * transfer_matrix[1][0] + g_i * transfer_matrix[1][1] + b_q * transfer_matrix[1][2]
    image_yiq_rgb[:, :, 2] = r_y * transfer_matrix[2][0] + g_i * transfer_matrix[2][1] + b_q * transfer_matrix[2][2]
    return image_yiq_rgb


def rgb2yiq(imRGB):
    """
    :param im_rgb: the rgb image
    :return: a new image with yiq
    """
    return rgb2yiq_yiq2rgb(imRGB, TRANSFER_MATRIX_YIQ_RGB)


def yiq2rgb(imYIQ):
    """
    :param im_yiq: the yiq image
    :return: the rgb image
    """
    inv_matrix = np.linalg.inv(TRANSFER_MATRIX_YIQ_RGB)
    return rgb2yiq_yiq2rgb(imYIQ, inv_matrix)


def isRGB(image):
    """
    :param img: given image, matrix of pixels
    :return: True if RGB, False if grayscale
    """
    if len(image.shape) == 3:
        return True
    else:
        return False


def stretch_hist(c_hist):
    '''

    :param c_hist: the image's histogram
    :return: stretch histogram
    '''
    first_nonzero = np.nonzero(c_hist)[0][0]
    mone = c_hist - c_hist[first_nonzero]
    mechane = (c_hist[-1] - c_hist[first_nonzero])
    norm_hist = ((mone * 255) / mechane).astype(np.int64)
    return norm_hist


def histogram_equalize(im_orig):
    """
    Performs histogram equalization on a given image,
    :param im_orig: the original image
    :return: the image after histogram eq, the original histogram and the new
    """
    if isRGB(im_orig):
        im_yiq = rgb2yiq(im_orig)
        im_orig_y = im_yiq[:, :, 0]
        im_orig_y = (im_orig_y * 255).astype(np.int64)
        # Step 1
        hist_im_origin = np.histogram(im_orig_y, bins=256, range=[0, 255])[0]
        # Step 2
        c_hist = np.cumsum(hist_im_origin)
        # Step 3 + 4
        c_hist = c_hist / c_hist[-1]
        # Step 5
        c_hist = stretch_hist(c_hist)
        c_hist = np.round(c_hist)
        eq_image = c_hist[im_orig_y]
        eq_image = eq_image.astype(np.int64)
        hist_eq = np.histogram(eq_image, bins=256, range=[0, 255])[0]
        im_yiq[:, :, 0] = eq_image / 255
        eq_image = yiq2rgb(im_yiq)
        return [eq_image, hist_im_origin, hist_eq]
    else:
        grey_image = (im_orig * 255).astype(np.int64)
        # Step 1
        hist_im_origin = np.histogram(grey_image,  bins=256, range=[0, 255])[0]
        # Step 2
        c_hist = hist_im_origin.cumsum()
        # Step 3 + 4
        c_hist_normal = c_hist / c_hist[-1]
        # Step 5
        c_hist_normal_stretch = stretch_hist(c_hist_normal)
        c_hist_normal_stretch = np.round(c_hist_normal_stretch)
        eq_image = c_hist_normal_stretch[grey_image]
        hist_eq = np.histogram(eq_image,  bins=256, range=[0, 255])[0]
        return [eq_image / 255, hist_im_origin, hist_eq]


def quantize_error(hist, z, q):
    """
    calculates the error
    :param hist: the image histogram
    :param z: the z array
    :param q: the q array
    :return: the error
    """
    error = 0
    for i in range(q.size - 1):
        array = np.arange(z[i] + 1, z[i+1] + 1)
        p = (q[i] - array) ** 2
        array = hist[z[i].astype(np.int64) + 1: z[i + 1].astype(np.int64) + 1]
        array = array * p
        error += array.sum()
    return error


def min_z(hist, z, n_iter, n_qaunt, q):
    """
    :param hist: image histogram
    :param n_quant: the number of q values
    :param n_iter: the number of iterations
    :return: the z array, q, error
    """
    errors = np.array([])
    arr_err = quantize_error(hist, z, q)
    errors = np.append(errors, arr_err)
    for i in range(1, n_iter):
        z_new = np.zeros(z.size)
        for i in range(1, n_qaunt):
            z_new[i] = (q[i - 1] + q[i]) // 2
        z_new[0] = 0
        z_new[-1] = 256
        find_q_of_z(hist, z_new, q)
        arr_err = quantize_error(hist, z_new, q)
        errors = np.append(errors, arr_err)
        z = z_new.copy()
    return z, q, errors


def init_z(c_hist, n_quant):
    """
    initial separaאק for the c_יhist
    :param c_hist: cumulative histogram of an image
    :param n_quant: number of quant
    :return: array
    """
    z = np.zeros(n_quant + 1)
    for i in range(1, n_quant):
        list_z = np.where(c_hist > (i * c_hist[-1] / n_quant))
        z[i] = list_z[0][0]
    z[-1] = 256
    return np.round(z)


def find_q_of_z(hist, z, q):
    for i in range(len(q)):
        weight = np.arange(z[i].astype(np.int64) , z[i + 1].astype(np.int64))
        q[i] = np.average(weight, weights=hist[z[i].astype(np.int64) : z[i + 1].astype(np.int64)])


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: input image
    :param n_quant: number of intensities the output im_quant should have
    :param n_iter: maximum number of iterations of the optimization procedure
    :return: list [im_quant,error] im_quant - is the quantized, error
    """
    im_orig_yiq = im_orig
    lut = np.zeros(256)
    if isRGB(im_orig):
        im_orig_yiq = rgb2yiq(im_orig)
        original_y = im_orig_yiq[:, :, 0]
    else:
        original_y = im_orig
    normalized = 255 * original_y
    hist = np.histogram(normalized, bins=257)[0]
    c_hist = np.cumsum(hist)
    z_of_q = init_z(c_hist, n_quant)
    q = np.zeros(n_quant)
    find_q_of_z(hist, z_of_q, q)
    z, q, err_arr = min_z(hist, z_of_q, n_iter, n_quant, q)
    for i in range(q.size):
        num_to_fill = q[i]
        begin = z[i].astype(np.int64)
        end = z[i + 1].astype(np.int64)
        lut[begin: end] = np.full_like(lut[begin: end], num_to_fill)
    im_new = lut[normalized.astype(np.int64)].astype(np.float32) / 255
    if isRGB(im_orig):
        im_orig_yiq[:, :, 0] = im_new
        im_new = yiq2rgb(im_orig_yiq)
    return [im_new, err_arr]