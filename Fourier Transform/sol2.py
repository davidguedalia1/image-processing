import numpy as np
import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from imageio import imread

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


def calculate_exp_signal(N, sign):
    '''
    Calculating the operator matrix of DFT.
    :param signal: Data
    :return: The DFT matrix
    '''

    arr_range = np.arange(N)
    u = arr_range.reshape(N, 1)
    element_wise_mut = np.multiply(u, arr_range)
    return np.exp((sign * 2 * np.pi * 1j * element_wise_mut) / N)


def DFT(signal):
    """
    The function gets a signal of 1 dimensional and
    return the DFT of the image.
    :param signal: Data
    :return: DFT of the signal
    """
    if signal.shape[0] == 0:
        return signal
    exp_signal = calculate_exp_signal(signal.shape[0], -1)
    signal_transpose = signal.T

    dft_signal = np.matmul(signal_transpose, exp_signal)
    return dft_signal.T


def IDFT(fourier_signal):
    """
    :param fourier_signal: fourier signal.
    :return: original signal.
    """
    if fourier_signal.shape[0] == 0:
        return fourier_signal
    exp_signal = calculate_exp_signal(fourier_signal.shape[0], 1) / fourier_signal.shape[0]
    signal_transpose = fourier_signal.T

    idft_signal = np.matmul(signal_transpose, exp_signal)
    return np.real_if_close(idft_signal.T)


def DFT2(image):
    """
    :param image: greyscale image.
    :return: fourier image.
    """
    dft2 = np.zeros(image.shape, dtype=np.complex128)
    for i in range(image.shape[0]):
        dft2[i, :] = DFT(image[i, :])
    for j in range(image.shape[1]):
        dft2[:, j] = DFT(dft2[:, j])
    return dft2


def IDFT2(fourier_image):
    """
    :param image: fourier image.
    :return: original image.
    """
    idft = np.zeros(fourier_image.shape, dtype=np.complex128)
    for i in range(fourier_image.shape[0]):
        idft[i, :] = IDFT(fourier_image[i, :])
    for j in range(fourier_image.shape[1]):
        idft[:, j] = IDFT(idft[:, j])
    return idft


def change_rate(filename, ratio):
    """
    Changes the duration of an audio file by keeping the same samples.
    :param filename:  WAV file
    :param ratio:change_samples representing the duration change.
    """
    rate, data = scipy.io.wavfile.read(filename)
    new_rate = int(ratio * rate)
    scipy.io.wavfile.write('change_rate.wav', new_rate, data)


def change_samples(filename, ratio):
    """
    Changes the duration of an audio file by reducing the number of samples
    using Fourier
    :param filename: is a string representing the path to a WAV file.
    :param ratio: representing the duration change.
    """
    rate, data = scipy.io.wavfile.read(filename)
    data = resize(data.astype(np.float64), ratio)
    scipy.io.wavfile.write('change_samples.wav', rate, data)
    return data


def resize(data, ratio):
    """
    :param data: data is a 1D ndarray of dtype float64  or complex128.
    :param ratio: representing the duration change.
    :return: value of resize is a 1D array of the dtype of data representing
    the new sample points.
    """
    if ratio == 1:
        return data
    data = DFT(data)
    data_size = data.size
    data = np.fft.fftshift(data)
    add_or_del = int(data_size // ratio)
    if add_or_del < data_size:
        d_data = data_size - add_or_del
        left = d_data // 2
        right = data_size + (d_data // -2)
        data_return = data[left: right]
    else:
        d_data = add_or_del - data_size
        left_len = d_data // 2
        right_len = -(d_data // -2)
        data_return = np.append(data, np.zeros(right_len))
        data_return = np.append(np.zeros(left_len), data_return)
    data_return = np.fft.ifftshift(data_return)
    data_return = IDFT(data_return)
    return np.real(data_return)


def resize_spectrogram(data, ratio):
    '''
    :param ratio: is a positive float64 representing the rate change of the WAV file.
    :param data: is a 1D ndarray of dtype float64 representing the original sample points.
    :return: new sample points according to ratio with the same datatype as data.
    '''
    stft_data = stft(data)
    new_data = []
    for vec in stft_data:
        new_element = resize(vec, ratio)
        new_data.append(new_element)
    new_data = np.array(new_data)
    new_data = istft(new_data)
    return new_data


def resize_vocoder(data, ratio):
    """
    Phase vocoding is the process of scaling the spectrogram as done before, but includes the correction of
    the phases of each frequency according to the shift of each window.
    :param ratio: is a positive float64 representing the rate change of the WAV file.
    :param data: is a 1D ndarray of dtype float64 representing the original sample points.
    :return: The function should return the given data rescaled according to ratio with the same datatype as data.
    """
    stft_data = stft(data)
    new_data = phase_vocoder(stft_data, ratio)
    return istft(new_data)


def conv_der(im):
    """
    Function that computes the magnitude of image derivatives.
    :param im: image
    :return:output is the magnitude of the derivative.
    """
    con_x = np.array([[0.5, 0, -0.5]])
    x = convolve2d(im, con_x, mode="same")
    con_y = con_x.T
    y = convolve2d(im, con_y, mode="same")
    magnitude = np.sqrt((np.abs(x) ** 2) + (np.abs(y) ** 2))
    return magnitude


def fourier_der(im):
    """
    Function that computes the magnitude of the image
    derivatives using Fourier transform.
    :param im: image.
    :return: magnitude of the image derivatives.
    """
    rows = im.shape[0]
    cols = im.shape[1]
    row_f = 2 * np.pi * 1j / rows
    cols_f = 2 * np.pi * 1j / cols
    rows_range = row_f * np.arange(int(-rows / 2), int(np.ceil(rows / 2))).reshape(rows, 1)
    cols_range = cols_f * np.arange(int(-cols / 2), int(np.ceil(cols / 2))).reshape(cols, 1)
    fourier = np.fft.fftshift(DFT2(im))
    mult_x = rows_range * fourier
    mult_y = cols_range * fourier.T
    dx = IDFT2(np.fft.ifftshift(mult_x))
    dy = IDFT2(np.fft.ifftshift(mult_y)).T
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase
    return warped_spec

def image_float(filename):
    '''
    :param filename: filename of the image
    :return: image
    '''
    image = imread(filename)
    image_to_return = image.astype(np.float64)
    image_to_return /= 255
    return image_to_return