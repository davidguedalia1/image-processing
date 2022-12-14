import imageio
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os

GRAY = 1


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function construct a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values.
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter.
    :return: pyr - array representing the pyramid, filter_vec - is a normalized row vector.
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    levels_pyr = min(max_levels - 1, len(gaussian_pyr) - 1)
    for i in range(levels_pyr):
        gaussian_expand = expand(gaussian_pyr[i + 1], filter_vec)
        sub_im = np.subtract(gaussian_pyr[i], gaussian_expand)
        pyr.append(sub_im)
    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def expand(im_layer, filter_vec):
    """
    Blur the image to expand it by two.
    :param im_layer: grayscale image with double values.
    :param filter_vec: row vector to blur.
    :return: blur_im - expand image.
    """
    rows, cols = im_layer.shape
    blur_im = np.zeros((rows * 2, cols * 2))
    blur_im[::2, ::2] = im_layer
    blur_im = ndimage.filters.convolve(blur_im, 2 * filter_vec)
    blur_im = ndimage.filters.convolve(blur_im, (2 * filter_vec).T)
    return blur_im


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Function that reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: array representing the pyramid
    :param filter_vec: The filter vector of 3.1
    :param coeff:  is a python list, length same as the number of levels in the pyramid lpyr.
    :return: image.
    """
    im_result = coeff[-1] * lpyr[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        expand_layer = expand(im_result, filter_vec)
        lpyr_coeff = coeff[i - 1] * lpyr[i - 1]
        im_result = expand_layer + lpyr_coeff
    return im_result


def render_pyramid(pyr, levels):
    """

    :param pyr: is either a Gaussian or Laplacian pyramid.
    :param levels: is the number of levels.
    :return: single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    pyr[0] = stretch_val(pyr[0])
    images = pyr[0]
    rows_length = pyr[0].shape[0]
    for i in range(1, levels):
        new_layer = stretch_val(pyr[i])
        new_layer.resize(rows_length, new_layer.shape[1])
        images = np.concatenate((images, new_layer), axis=1)
    return images


def stretch_val(pyr):
    return (pyr - pyr.min()) / (pyr.max() - pyr.min())


def display_pyramid(pyr, levels):
    plt.figure()
    plt.imshow(render_pyramid(pyr, levels), cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Function that do pyramid blending.
    :param im1: grayscale images to be blended
    :param im2: grayscale images to be blended
    :param mask: – is a boolean mask containing True and False representing which parts
               of im1 and im2 should appear in the resulting im_blend.
    :param max_levels:  is the max_levels parameter you should use when generating the Gaussian and Laplacian.
    :param filter_size_im: defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: image blend.
    """
    l_1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l_2, ignore = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_m, ignore = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for k in range(len(l_1)):
        l_out.append(g_m[k] * l_1[k] + (1 - g_m[k]) * l_2[k])
    coeff = [1] * len(l_out)
    l_out = laplacian_to_image(l_out, filter_vec, coeff)
    return np.clip(l_out, 0, 1)


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


def blending_example1():
    im1 = read_image(relpath('externals/casino_table.jpg'), 2)
    im2 = read_image(relpath('externals/panel_friday.jpg'), 2)
    mask = read_image(relpath('externals/mask_panel.jpg'), 1)
    mask = mask.round().astype(bool)
    res = blend(im1, im2, mask)
    return im1, im2, mask, res


def blending_example2():
    im1 = read_image(relpath('externals/chanuka.jpg'), 2)
    im2 = read_image(relpath('externals/skuid_chanuka.jpg'), 2)
    mask = read_image(relpath('externals/mask_chanuka.jpg'), 1)
    mask = mask.round().astype(bool)
    res = blend(im1, im2, mask)
    return im1, im2, mask, res


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blend(im1, im2, mask):
    r = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 3, 6, 6)
    g = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 3, 6, 6)
    b = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 3, 6, 6)
    blend_image = np.dstack((r, g, b))
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(im1, cmap='gray')
    ax[0][1].imshow(im2, cmap='gray')
    ax[1][0].imshow(mask, cmap='gray')
    ax[1][1].imshow(blend_image, cmap='gray')
    plt.show()
    return blend_image
