import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from templatetracker.correlationfilter.utils import pad, crop, fast2dconv


def gaussian_correlation(x, z, sigma=0.2, boundary='constant'):
    r"""
    Gaussian kernel correlation.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Template image.
    z : ``(channels, height, width)`` `ndarray`
        Input image.
    sigma: `float`, optional
        Kernel std.

    Returns
    -------
    xz: ``(1, height, width)`` `ndarray`
        Gaussian kernel correlation between the image and the template.
    """
    # norms
    x_norm = x.ravel().T.dot(x.ravel())
    z_norm = z.ravel().T.dot(z.ravel())
    # cross correlation
    xz = np.sum(fast2dconv(z, x[:, ::-1, ::-1], boundary=boundary), axis=0)
    # gaussian kernel
    kxz = np.exp(-(1/sigma**2) * np.maximum(0, x_norm + z_norm - 2 * xz))
    return kxz[None]


def polynomial_correlation(x, z, a=5, b=1, boundary='constant'):
    r"""
    Polynomial kernel correlation.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Template image.
    z : ``(channels, height, width)`` `ndarray`
        Input image.
    a: `float`, optional
        Kernel exponent.
    b: `float`, optional
        Kernel constant.

    Returns
    -------
    kxz: ``(1, height, width)`` `ndarray`
        Polynomial kernel correlation between the image and the template.
    """
    # cross correlation
    xz = np.sum(fast2dconv(z, x[:, ::-1, ::-1], boundary=boundary), axis=0)
    # polynomial kernel
    kxz = (xz + b) ** a
    return kxz[None]


def linear_correlation(x, z, boundary='constant'):
    r"""
    Linear kernel correlation (dot product).

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Template image.
    z : ``(channels, height, width)`` `ndarray`
        Input image.

    Returns
    -------
    xz: ``(1, height, width)`` `ndarray`
        Linear kernel correlation between the image and the template.
    """
    # cross correlation
    xz = np.sum(fast2dconv(z, x[:, ::-1, ::-1], boundary=boundary), axis=0)
    return xz[None]


def learn_kcf(x, y, kernel_correlation=gaussian_correlation, l=0.01,
              boundary='constant', **kwargs):
    r"""
    Kernelized Correlation Filter (KCF).

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Template image.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    kernel_correlation: `callable`, optional
        Callable implementing a particular type of kernel correlation i.e.
        gaussian, polynomial or linear.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    alpha: ``(channels, height, width)`` `ndarray`
        Kernelized Correlation Filter (KFC) associated to the template image.

    References
    ----------
    .. [1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista. "High-Speed
    Tracking with Kernelized Correlation Filters". TPAMI, 2015.
    """
    # extended shape
    x_shape = np.asarray(x.shape[-2:])
    y_shape = np.asarray(y.shape[-2:])
    ext_shape = x_shape + y_shape - 1
    # extend desired response
    ext_x = pad(x, ext_shape, boundary=boundary)
    ext_y = pad(y, ext_shape)
    # ffts of extended auto kernel correlation and extended desired response
    fft_ext_kxx = fft2(kernel_correlation(ext_x, ext_x, **kwargs))
    fft_ext_y = fft2(ext_y)
    # extended kernelized correlation filter
    ext_alpha = np.real(ifftshift(ifft2(fft_ext_y / (fft_ext_kxx + l)),
                                  axes=(-2, -1)))
    return crop(ext_alpha, y_shape), crop(x, y_shape)


def learn_deep_kcf(x, y, n_levels=3, kernel_correlation=gaussian_correlation,
                   l=0.01, boundary='constant', **kwargs):
    r"""
    Deep Kernelized Correlation Filter (DKCF).

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Template image.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    n_levels: `int`, optional
        Number of levels.
    kernel_correlation: `callable`, optional
        Callable implementing a particular type of kernel correlation i.e.
        gaussian, polynomial or linear.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    deep_alpha: ``(channels, height, width)`` `ndarray`
        Deep Kernelized Correlation Filter (DKFC), in the frequency domain,
        associated to the template image.
    """
    # learn alpha
    alpha = learn_kcf(x, y, kernel_correlation=kernel_correlation, l=l,
                      boundary=boundary, **kwargs)

    # initialize alphas
    alphas = np.empty((n_levels,) + alpha.shape)
    # for each level
    for l in range(n_levels):
        # store filter
        alphas[l] = alpha
        # compute kernel correlation between template and image
        kxz = kernel_correlation(x, x, **kwargs)
        # compute kernel correlation response
        x = fast2dconv(kxz, alpha)
        # learn mosse filter from responses
        alpha = learn_kcf(x, y, kernel_correlation=kernel_correlation, l=l,
                          boundary=boundary, **kwargs)

    # compute equivalent deep mosse filter
    deep_alpha = alphas[0]
    for a in alphas[1:]:
        deep_alpha = fast2dconv(a, a, boundary=boundary)

    return deep_alpha, alphas