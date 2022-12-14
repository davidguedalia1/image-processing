from scipy.signal import convolve2d
import numpy as np
from skimage.color import rgb2gray
import imageio
from scipy import ndimage

GRAY = 1


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


def image_float(filename):
    '''
    :param filename: filename of the image
    :return: image
    '''
    image = imageio.imread(filename)
    image_to_return = image.astype(np.float64)
    image_to_return /= 255
    return image_to_return

def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function construct a Gaussian pyramid of a given image.
    :param im: a grayscale image with double values.
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: pyr - array representing the pyramid, filter_vec - is a normalized row vector.
     """
    filter_vec = create_filter(filter_size)
    i = 1
    pyr = [im]
    cur_im = im
    while i < max_levels and cur_im.shape > (16, 16):
        cur_im = blur_reduce(filter_vec, pyr)
        i += 1
    return pyr, filter_vec


def create_filter(filter_size):
    """
    Calculates the filter.
    :param filter_size: how many convolutions needed
    :return: filter vector
    """
    if filter_size == 1:
        return np.array([[1]])
    else:
        con = np.array([[1, 1]])
        filter_vec = np.array([[1, 1]])
        for i in range(1, filter_size - 1):
            filter_vec = convolve2d(con, filter_vec)
        sum_filter = np.sum(filter_vec)
        filter_vec = filter_vec / sum_filter
    return filter_vec.astype(np.float64)


def blur_reduce(filter_vec, pyr):
    """
    Blurring and reducing the im - add it to pyr.
    :param filter_size: the size of the Gaussian filter.
    :param pyr: array representing the pyramid, filter_vec - is a normalized row vector.
    """
    im = pyr[-1]
    blur_im = ndimage.filters.convolve(im, filter_vec)
    blur_im = ndimage.filters.convolve(blur_im, filter_vec.T)
    reduce_im = blur_im[::2, ::2]
    pyr.append(reduce_im)
    return reduce_im




