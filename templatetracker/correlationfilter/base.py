import numpy as np
from menpo.shape import PointCloud
from menpo.transform import Affine
from menpo.feature import no_op

from templatetracker.correlationfilter.correlationfilter import (
    learn_mosse, increment_mosse)
from templatetracker.correlationfilter.kernelizedfilter import (
    learn_kcf, gaussian_correlation)
from .utils import (
    build_grid, generate_gaussian_response, normalizenorm_vec, fast2dconv,
    compute_psr)


def extract_targets(frame, target_centre, target_shape, n_perturbations=10,
                    noise_std=0.04):
    # initialize targets
    w, h = target_shape
    targets = np.empty((n_perturbations + 1, frame.n_channels, w, h))

    # extract original target
    targets[0] = frame.extract_patches(
        target_centre, patch_size=target_shape,
        as_single_array=True)

    for j in range(n_perturbations):
        # perturb identity affine transform
        params = noise_std * np.random.randn(6)
        transform = Affine.init_identity(2).from_vector(params)
        # warp frame using previous affine transform
        perturbed_frame = frame.warp_to_mask(frame.as_masked().mask,
                                             transform)
        # apply inverse of affine transform to target centre
        perturbed_centre = transform.pseudoinverse().apply(target_centre)
        # extract perturbed target + context region from frame
        perturbed_target = perturbed_frame.extract_patches(
            perturbed_centre, patch_size=target_shape,
            as_single_array=True)
        # store target
        targets[j+1] = perturbed_target

    return targets


def compute_max_peak(response):
    corrector = np.asarray(response.shape[1:]) // 2 + 1
    max_index = np.unravel_index(response.argmax(), response.shape[1:])
    offset = np.asarray(max_index, dtype=np.double) - corrector
    return offset


def compute_meanshift_peak(response, cov=10):
    # obtain all possible offsets
    offsets = build_grid(response.shape)
    # turn response into pseudo-likelihood
    response = response - np.min(response)
    response = response * generate_gaussian_response(response.shape[1:], cov)
    response = response / np.sum(response)
    # compute mean shift offset by multiplying all offsets by their likelihood
    offset = np.sum(response * np.rollaxis(offsets, -1), axis=(-2, -1))
    return offset


class CFTracker():

    def __init__(self, frame, target_centre, target_shape, context=2,
                 learn_filter=learn_mosse, increment_filter=increment_mosse,
                 features=no_op, response_cov=3, n_perturbations=10,
                 noise_std=0.04, l=0.01, normalize=normalizenorm_vec,
                 mask=True, boundary='constant'):
        self.initialize(frame, target_centre, target_shape, context=context,
                        learn_filter=learn_filter,
                        increment_filter=increment_filter, features=features,
                        response_cov=response_cov,
                        n_perturbations=n_perturbations, noise_std=noise_std,
                        l=l, normalize=normalize, mask=mask, boundary=boundary)

    def _preprocess_vec(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        if self.cosine_mask is not None:
            x = self.cosine_mask * x
        return x

    def initialize(self, frame, target_centre, target_shape, context=2,
                   features=no_op, learn_filter=learn_mosse,
                   increment_filter=increment_mosse, response_cov=3,
                   n_perturbations=10, noise_std=0.04, l=0.01,
                   normalize=normalizenorm_vec, mask=True,
                   boundary='constant'):

        self.target_shape = target_shape
        self.learn_filter = learn_filter
        self.increment_filter = increment_filter
        self.features = features
        self.l = l
        self.normalize = normalize
        self.boundary = boundary

        # compute context shape
        self.context_shape = np.round(context * np.asarray(target_shape))
        self.context_shape[0] += (0 if np.remainder(self.context_shape[0], 2)
                                  else 1)
        self.context_shape[1] += (0 if np.remainder(self.context_shape[1], 2)
                                  else 1)

        # compute subframe size
        self.subframe_shape = self.context_shape + 8
        # compute target centre coordinates in subframe
        self.subframe_target_centre = PointCloud((
            self.subframe_shape // 2)[None])

        # extract subframe
        subframe = frame.extract_patches(target_centre,
                                         patch_size=self.subframe_shape)[0]

        # compute features
        subframe = self.features(subframe)

        # obtain targets
        targets = extract_targets(subframe, self.subframe_target_centre,
                                  self.context_shape, n_perturbations,
                                  noise_std)

        # generate gaussian response
        self.response = generate_gaussian_response(self.target_shape[-2:],
                                                   response_cov)

        if mask:
            cy = np.hanning(self.context_shape[0])
            cx = np.hanning(self.context_shape[1])
            self.cosine_mask = cy[..., None].dot(cx[None, ...])

        targets_pp = []
        for j, t in enumerate(targets):
            targets_pp.append(self._preprocess_vec(t))
        targets_pp = np.asarray(targets_pp)

        self.filter, self.num, self.den = self.learn_filter(
            targets_pp, self.response, l=self.l, boundary=self.boundary)

    def track(self, frame, target_centre, nu=0.05, psr_threshold=10,
              compute_peak=compute_max_peak):
        # extract subframe
        subframe = frame.extract_patches(target_centre,
                                         patch_size=self.subframe_shape)[0]
        # compute features
        subframe = self.features(subframe)

        # extract surrounding region around previous target centre
        x = subframe.extract_patches(self.subframe_target_centre,
                                     patch_size=self.context_shape)[0].pixels

        # normalize surrounding region if needed
        if self.normalize is not None:
            x = self.normalize(x)
        # compute response
        response = np.sum(fast2dconv(x, self.filter,
                                     boundary=self.boundary), axis=0)[None]
        # update target centre
        peak = compute_peak(response)
        target_centre = PointCloud(target_centre.points + peak)

        # compute peak to sidelobe ratio
        psr = compute_psr(response)
        if psr > psr_threshold:
            # extract new target
            target = frame.extract_patches(
                target_centre, patch_size=self.context_shape)[0]
            # compute features
            target = self.features(target).pixels
            # pre-process new target to match learning conditions
            target = self._preprocess_vec(target)
            # learn filter associate to new target
            self.filter, self.num, self.den = self.increment_filter(
                self.num, self.den, target[None], self.response, l=self.l,
                boundary=self.boundary)

        return target_centre, psr, response


class KCFTracker():

    def __init__(self, frame, target_centre, target_shape, context=2,
                 learn_filter=learn_kcf,
                 kernel_correlation=gaussian_correlation,
                 features=no_op, response_cov=3, l=0.01,
                 normalize=normalizenorm_vec, mask=True,
                 boundary='constant', **kwargs):
        self.initialize(frame, target_centre, target_shape, context=context,
                        learn_filter=learn_filter,
                        kernel_correlation=kernel_correlation,
                        features=features, response_cov=response_cov,
                        l=l, normalize=normalize, mask=mask,
                        boundary=boundary, **kwargs)

    def _preprocess_vec(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        if self.cosine_mask is not None:
            x = self.cosine_mask * x
        return x

    def initialize(self, frame, target_centre, target_shape, context=2,
                   features=no_op, learn_filter=learn_kcf,
                   kernel_correlation=gaussian_correlation, response_cov=3,
                   l=0.01, normalize=normalizenorm_vec, mask=True,
                   boundary='constant', **kwargs):

        self.target_shape = target_shape
        self.learn_filter = learn_filter
        self.kernel_correlation = kernel_correlation
        self.features = features
        self.l = l
        self.normalize = normalize
        self.boundary = boundary

        # compute context shape
        self.context_shape = np.round(context * np.asarray(target_shape))
        self.context_shape[0] += (0 if np.remainder(self.context_shape[0], 2)
                                  else 1)
        self.context_shape[1] += (0 if np.remainder(self.context_shape[1], 2)
                                  else 1)

        # compute subframe size
        self.subframe_shape = self.context_shape + 8
        # compute target centre coordinates in subframe
        self.subframe_target_centre = PointCloud((
            self.subframe_shape // 2)[None])

        # extract subframe
        subframe = frame.extract_patches(target_centre,
                                         patch_size=self.subframe_shape)[0]

        # compute features
        subframe = self.features(subframe)

        # obtain targets
        target = subframe.extract_patches(self.subframe_target_centre,
                                          patch_size=self.context_shape)[0]

        # generate gaussian response
        self.response = generate_gaussian_response(self.target_shape[-2:],
                                                   response_cov)

        if mask:
            cy = np.hanning(self.context_shape[0])
            cx = np.hanning(self.context_shape[1])
            self.cosine_mask = cy[..., None].dot(cx[None, ...])

        target = self._preprocess_vec(target.pixels)

        self.alpha, self.target = self.learn_filter(
            target, self.response, kernel_correlation=self.kernel_correlation,
            l=self.l, boundary=self.boundary, **kwargs)

    def track(self, frame, target_centre, nu=0.05, psr_threshold=10,
              compute_peak=compute_max_peak, **kwargs):
        # extract subframe
        subframe = frame.extract_patches(target_centre,
                                         patch_size=self.subframe_shape)[0]
        # compute features
        subframe = self.features(subframe)

        # extract surrounding region around previous target centre
        z = subframe.extract_patches(self.subframe_target_centre,
                                     patch_size=self.context_shape)[0].pixels

        # normalize surrounding region if needed
        x = self.target
        if self.normalize is not None:
            z = self.normalize(z)
            x = self.normalize(x)

        # compute kernel correlation between template and image
        kxz = self.kernel_correlation(x, z, boundary=self.boundary, **kwargs)
        # compute kernel correlation response
        response = fast2dconv(kxz, self.alpha, boundary=self.boundary)

        # update target centre
        peak = compute_peak(response)
        target_centre = PointCloud(target_centre.points + peak)

        # compute peak to sidelobe ratio
        psr = compute_psr(response)
        if psr > psr_threshold:
            # extract new target
            target = frame.extract_patches(
                target_centre, patch_size=self.context_shape)[0]
            # compute features
            target = self.features(target).pixels
            # pre-process new target to match learning conditions
            target = self._preprocess_vec(target)
            # learn filter associate to new target
            alpha, target = self.learn_filter(
                target, self.response,
                kernel_correlation=self.kernel_correlation, l=self.l,
                boundary=self.boundary, **kwargs)
            # update filter
            self.alpha = (1 - nu) * self.alpha + nu * alpha
            self.target = (1 - nu) * self.target + nu * target

        return target_centre, psr, response