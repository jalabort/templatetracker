import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve

from templatetracker.correlationfilter.utils import pad, crop, fast2dconv


def learn_mosse(X, y, l=0.01, boundary='constant'):
    r"""
    Minimum Output Sum od Squared Errors (MOSSE) filter.

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    mosse: ``(1, height, width)`` `ndarray`
        Minimum Output Sum od Squared Errors (MOSSE) filter associated to
        the training images.

    References
    ----------
    .. [1] David S. Bolme, J. Ross Beveridge,  Bruce A. Draper and Yui Man Lui.
    "High-Speed Tracking with Kernelized Correlation Filters". CVPR, 2010.
    """
    # number of images, number of channels, height and width
    n, k, hx, wx = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for x in X:
        # extend image
        ext_x = pad(x, ext_shape, boundary=boundary)
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # update auto and cross spectral energy matrices
        sXX += fft_ext_x.conj() * fft_ext_x
        sXY += fft_ext_x.conj() * fft_ext_y

    # compute desired correlation filter
    fft_ext_f = sXY / (sXX + l)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    ext_f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))
    # crop extended filter to match desired response shape
    f = crop(ext_f, y_shape)

    return f, sXY, sXX


def increment_mosse(A, B, X, y, nu=0.5, l=0.01, boundary='constant'):
    r"""
    Incremental Minimum Output Sum od Squared Errors (iMOSSE) filter
    """
    # number of images, number of channels, height and width
    n, k, hx, wx = X.shape
    x_shape = (hx, wx)

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for x in X:
        # extend image
        ext_x = pad(x, ext_shape, boundary=boundary)
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # update auto and cross spectral energy matrices
        sXX += fft_ext_x.conj() * fft_ext_x
        sXY += fft_ext_x.conj() * fft_ext_y

    # combine old and new auto and cross spectral energy matrices
    sXY = (1 - nu) * A + nu * sXY
    sXX = (1 - nu) * B + nu * sXX
    # compute desired correlation filter
    fft_ext_f = sXY / (sXX + l)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    ext_f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))
    # crop extended filter to match desired response shape
    f = crop(ext_f, y_shape)

    return f, sXY, sXX


def learn_mccf(X, y, l=0.01, boundary='constant'):
    r"""
    Multi-Channel Correlation Filter (MCCF).

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    mccf: ``(1, height, width)`` `ndarray`
        Multi-Channel Correlation Filter (MCCF) filter associated to the
        training images.

    References
    ----------
    .. [1] Hamed Kiani Galoogahi, Terence Sim,  Simon Lucey. "Multi-Channel
    Correlation Filters". ICCV, 2013.
    """
    # number of images; number of channels, height and width
    n, k, hx, wx = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)
    # extended dimensionality
    ext_d = ext_h * ext_w

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for x in X:
        # extend image
        ext_x = pad(x, ext_shape, boundary=boundary)
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # store extended image fft as sparse diagonal matrix
        diag_fft_x = spdiags(fft_ext_x.reshape((k, -1)),
                             -np.arange(0, k) * ext_d, ext_d * k, ext_d).T
        # vectorize extended desired response fft
        diag_fft_y = fft_ext_y.ravel()

        # update auto and cross spectral energy matrices
        sXX += diag_fft_x.conj().T.dot(diag_fft_x)
        sXY += diag_fft_x.conj().T.dot(diag_fft_y)

    # solve ext_d independent k x k linear systems (with regularization)
    # to obtain desired extended multi-channel correlation filter
    fft_ext_f = spsolve(sXX + l * speye(sXX.shape[-1]), sXY)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    ext_f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))
    # crop extended filter to match desired response shape
    f = crop(ext_f, y_shape)

    return f, sXY, sXX


def increment_mccf(A, B, X, y, nu=0.125, l=0.01, boundary='constant'):
    r"""
    Incremental Multi-Channel Correlation Filter (MCCF)
    """
    # number of images; number of channels, height and width
    n, k, hx, wx = X.shape
    x_shape = (hx, wx)

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)
    # extended dimensionality
    ext_d = ext_h * ext_w

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for x in X:
        # extend image
        ext_x = pad(x, ext_shape, boundary=boundary)
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # store extended image fft as sparse diagonal matrix
        diag_fft_x = spdiags(fft_ext_x.reshape((k, -1)),
                             -np.arange(0, k) * ext_d, ext_d * k, ext_d).T
        # vectorize extended desired response fft
        diag_fft_y = fft_ext_y.ravel()

        # update auto and cross spectral energy matrices
        sXX += diag_fft_x.conj().T.dot(diag_fft_x)
        sXY += diag_fft_x.conj().T.dot(diag_fft_y)

    # combine old and new auto and cross spectral energy matrices
    sXY = (1 - nu) * A + nu * sXY
    sXX = (1 - nu) * B + nu * sXX
    # solve ext_d independent k x k linear systems (with regularization)
    # to obtain desired extended multi-channel correlation filter
    fft_ext_f = spsolve(sXX + l * speye(sXX.shape[-1]), sXY)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    ext_f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))
    # crop extended filter to match desired response shape
    f = crop(ext_f, y_shape)

    return f, sXY, sXX


def learn_deep_cf(X, y, learn_cf=learn_mccf, n_levels=3, l=0.01,
                  boundary='constant'):
    r"""
    Deep Correlation Filter (DCF) filter.

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    learn_cf : `callable`, optional
        Callable used to learn a single shallow Correlation Filter (CF) (
        `learn_mosse` for single channel images or `learn_mccf` for
        multi-channel images).
    n_levels : int, optional
        The number of level.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    deep_mosse: ``(1, height, width)`` `ndarray`
        Deep Correlation Filter (DCF) filter associated to the training images.
    """
    # learn mosse filter
    f, A, B = learn_cf(X, y, l=l, boundary=boundary)

    # initialize filters, numerators and denominators arrays
    fs = np.empty((n_levels,) + f.shape)
    As = [A]
    Bs = [B]
    # store filter
    fs[0] = f
    # for each level
    for k in range(1, n_levels):
        nX = []
        # for each training image
        for j, x in enumerate(X):
            # convolve image and filter
            x = np.sum(fast2dconv(x, f, mode='full', boundary=boundary),
                       axis=0)[None]
            # replace image with response
            nX.append(x)
        X = np.asarray(nX)
        # learn mosse filter from responses
        f, A, B = learn_cf(X, y, l=l, boundary=boundary)
        # store filter, numerator and denominator
        fs[k] = f
        As.append(A)
        Bs.append(B)

    # compute equivalent deep cf filter
    df = fs[0]
    for f in fs[1:]:
        df = fast2dconv(df, f, boundary=boundary)

    return df, As, Bs


def increment_deep_cf(As, Bs, X, y, increment_cf=increment_mccf,
                      n_levels=3, l=0.01, boundary='constant'):
    r"""
    Increment Deep Correlation Filter (DCF) filter.

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    learn_cf : `callable`, optional
        Callable used to learn a single shallow Correlation Filter (CF) (
        `learn_mosse` for single channel images or `learn_mccf` for
        multi-channel images).
    n_levels : int, optional
        The number of level.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.

    Returns
    -------
    deep_mosse: ``(1, height, width)`` `ndarray`
        Deep Correlation Filter (DCF) filter associated to the training images.
    """
    # learn mosse filter
    f, A, B = increment_cf(As[0], Bs[0], X, y, l=l, boundary=boundary)

    # initialize filters, numerators and denominators arrays
    fs = np.empty((n_levels,) + f.shape)
    As[0] = A
    Bs[0] = B
    # store filter
    fs[0] = f
    # for each level
    for k in range(1, n_levels):
        nX = []
        # for each training image
        for j, x in enumerate(X):
            # convolve image and filter
            x = np.sum(fast2dconv(x, f, mode='full', boundary=boundary),
                       axis=0)[None]
            # replace image with response
            nX.append(x)
        X = np.asarray(nX)
        # learn mosse filter from responses
        f, A, B = increment_cf(As[k], Bs[k], X, y, l=l, boundary=boundary)
        # store filter, numerator and denominator
        fs[k] = f
        As[k] = A
        Bs[k] = B

    # compute equivalent deep mosse filter
    df = fs[0]
    for f in fs[1:]:
        df = fast2dconv(df, f, boundary=boundary)

    return df, As, Bs