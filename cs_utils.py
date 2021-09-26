import numpy as np

def fourier_coeffs(block, num_retain_coeff):
    """
    Creates coefficients retaining top k coefficients of the block image.

    :param block: block image
    :type block: numpy.ndarray
    :param num_retain_coeff: the number of top coefficients to retain
    :type num_retain_coeff: int

    :return: coefficients
    """

    # convert the image to a single column vector
    img_vector = block.ravel()

    # compute the FFT
    fourier = np.fft.fft(img_vector)

    # retain the top 'k' coefficients, and zero out the remaining ones
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier).ravel())
    coefficients = fourier
    coefficients[sorted_indices[num_retain_coeff:]] = 0.0

    return coefficients
