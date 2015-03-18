import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from scipy.stats import multivariate_normal

from menpo.shape import PointDirectedGraph


def pad(pixels, ext_shape, boundary='constant'):
    _, h, w = pixels.shape

    h_margin = (ext_shape[0] - h) // 2
    w_margin = (ext_shape[1] - w) // 2

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[0]:
        h_margin += 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[1]:
        w_margin += 1

    pad_width = ((0, 0), (h_margin, h_margin2), (w_margin, w_margin2))

    return np.lib.pad(pixels, pad_width, mode=boundary)


def crop(pixels, shape):
    _, h, w = pixels.shape

    h_margin = (h - shape[0]) // 2
    w_margin = (w - shape[1]) // 2

    h_corrector = 1 if np.remainder(h - shape[0], 2) != 0 else 0
    w_corrector = 1 if np.remainder(w - shape[1], 2) != 0 else 0

    return pixels[:,
                  h_margin + h_corrector:-h_margin,
                  w_margin + w_corrector:-w_margin]


def fast2dconv(x, f, mode='same', boundary='constant'):
    r"""
    Performs fast 2d convolution in the frequency domain.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Image.
    f : ``(channels, height, width)`` `ndarray`
        Filter.
    mode : str {`full`, `same`, `valid`}, optional
        Determines the shape of the resulting convolution.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    Returns
    -------
    c: ``(channels, height, width)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel.
    """
    # extended shape
    x_shape = np.asarray(x.shape[-2:])
    f_shape = np.asarray(f.shape[-2:])
    ext_shape = x_shape + f_shape - 1

    # extend image and filter
    ext_x = pad(x, ext_shape, boundary=boundary)
    ext_f = pad(f, ext_shape)

    # compute ffts of extended image and extended filter
    fft_ext_x = fft2(ext_x)
    fft_ext_f = fft2(ext_f)

    # compute extended convolution in Fourier domain
    fft_ext_c = fft_ext_f * fft_ext_x

    # compute ifft of extended convolution
    ext_c = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, x_shape)
    elif mode is 'valid':
        return crop(ext_c, x_shape - f_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))


def compute_psr(response, neighbours_shape=(11, 11)):
    r"""
    Peak to Sidelobe Ratio (PSR).

    Parameters
    ----------
    response : ``(1, height, width)`` `ndarray`
        response image.
    neighbours_shape : ``(h, w)`` `ndarray`
        Shape determining the amount of neighbours around the peak
        (including the peak) that are not considered part of the sidelobe.

    Returns
    -------
    psr: `float`
        Peak to Sidelobe Ratio (PSR).

    References
    ----------
    .. [1] David S. Bolme, J. Ross Beveridge,  Bruce A. Draper and Yui Man Lui.
    "High-Speed Tracking with Kernelized Correlation Filters". CVPR, 2010.
    """
    response = response[0]
    response_shape = response.shape
    h, w = response_shape

    peak = np.unravel_index(response.argmax(), response_shape)

    neighbours_grid = build_grid(neighbours_shape)
    neighbours_indices = peak + neighbours_grid
    y = neighbours_indices[:, 0]
    x = neighbours_indices[:, 1]

    y = y[y < h]
    y = y[y > 0]
    x = x[x < w]
    x = x[x > 0]
    y, x = np.meshgrid(y, x)
    neighbours_indices = np.vstack((y[:], x[:]))

    boolean_indices = np.ones_like(response, dtype=np.bool)
    boolean_indices[neighbours_indices] = False

    sidelobe = response[boolean_indices]
    mu = np.mean(sidelobe)
    std = np.std(sidelobe)

    return (response[peak] - mu) / std


def build_grid(shape):
    shape = np.array(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    start = -half_shape
    end = half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)


def centralize_vec(x):
    centred_x = x - np.mean(x)
    return centred_x


def normalizenorm_vec(x):
    centred_x = centralize_vec(x)
    return centred_x / np.linalg.norm(centred_x)


def normalizestd_vec(x):
    centred_x = centralize_vec(x)
    return centred_x / np.std(centred_x)


def generate_gaussian_response(shape, cov):
    mvn = multivariate_normal(mean=np.zeros(2), cov=cov)
    grid = build_grid(shape)
    return mvn.pdf(grid)[None]


def generate_bounding_box(target_centre, target_size):
    y, x = np.asarray(target_size) / 2
    p0 = target_centre.points + [-y, -x]
    p1 = target_centre.points + [-y,  x]
    p2 = target_centre.points + [ y,  x]
    p3 = target_centre.points + [ y, -x]
    return PointDirectedGraph(np.vstack((p0, p1, p2, p3)),
                              np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
                              copy=False)
