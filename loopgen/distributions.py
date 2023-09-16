from typing import Union, Tuple, Any

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from math import pi
from abc import ABC, abstractmethod
from e3nn.o3 import axis_angle_to_matrix

from .utils import is_positive_semidefinite, is_symmetric


SampleSize = Union[int, Tuple[int, ...]]


class Distribution(ABC):

    """
    Abstract class for representing a probability distribution, with methods `pdf()` and `sample()`.
    """

    @abstractmethod
    def __init__(self, *params: Any, **kw_params: Any):
        """
        Initialises an instance of the distribution with some parameters.
        """

        pass

    @abstractmethod
    def pdf(self, value: Any):
        """
        Calculates the probability density for some value(s) under the parametrised distribution.
        """

        pass

    @abstractmethod
    def sample(
        self, size: SampleSize, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Samples some values from the distribution with the specified size, returning a tensor.
        """

        pass


class Gaussian3DDistribution(Distribution):

    """
    Standard multivariate gaussian distribution for 3 dimensions,
    using the PyTorch implementation `MultivariateNormal`
    but with the `Distribution` interface defined here.
    """

    def __init__(self, mean: torch.Tensor, var: Union[torch.Tensor, float] = 1.0):
        if isinstance(mean, torch.Tensor) and mean.shape == (3,):
            self._mean = mean
        else:
            raise ValueError(
                "Argument mean should either be a float or a tensor of shape (3,)"
            )

        if isinstance(var, float):
            self._cov_matrix = torch.zeros((3, 3), device=mean.device)
            self._cov_matrix.fill_diagonal_(var)
            self._prec_matrix = 1 / self._cov_matrix
            self._is_isotropic = True
        elif isinstance(var, torch.Tensor) and var.shape == (3, 3):
            assert is_positive_semidefinite(
                var
            ), "Covariance must be positive semidefinite"
            assert is_symmetric(var), "Covariance matrix must be symmetric"
            self._cov_matrix = var
            self._prec_matrix = torch.linalg.inv(self._cov_matrix)
            self._is_isotropic = False
        else:
            raise ValueError(
                "Argument var should either be a float or a tensor of shape (3, 3)"
            )

        self._distribution = MultivariateNormal(
            loc=self._mean, covariance_matrix=self._cov_matrix
        )

    @property
    def mean(self) -> torch.Tensor:
        """The mean of the distribution."""
        return self._mean

    @property
    def cov_matrix(self) -> torch.Tensor:
        """The covariance matrix of the distribution."""
        return self._cov_matrix

    @property
    def prec_matrix(self) -> torch.Tensor:
        """The precision matrix of the distribution."""
        return self._prec_matrix

    @property
    def is_isotropic(self) -> bool:
        """
        Whether the distribution has a scalar variance
        (i.e. a diagonal covariance matrix with constant value) or not.
        """
        return self._is_isotropic

    def pdf(self, value: torch.Tensor) -> torch.Tensor:
        """The probability density function of the distribution."""
        return torch.exp(self._distribution.log_prob(value))

    def sample(
        self, size: SampleSize, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Samples from the distribution."""
        if isinstance(size, int):
            size = (size,)

        return self._distribution.sample(size).to(device)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={self._mean}, "
            f"cov={self._cov_matrix})"
        )


class IGSO3Distribution(Distribution):

    """
    Contains code for the isotropic gaussian distribution on SO(3), which can be used to sample rotations.

    The distribution takes a single variance parameter `var` that determines the width of the distribution
    "around" the identity matrix. Sampling is performed via inverse-transform sampling using an interpolated
    approximation of the inverse CDF. Just call the `sample()` method, which takes in a shape (either an int or
    a tuple of ints) as its only argument and generates a set of random rotations.
    """

    def __init__(
        self,
        var: float = 1.0,
        support_n: int = 5000,
        expansion_n: int = 5000,
        as_matrices: bool = True,
        uniform: bool = False,
    ):
        assert var > 0, "Variance must be greater than 0"
        assert (
            support_n > 0 and expansion_n > 0
        ), "Length of support and expansion must be greater than 0"

        self._var = var
        self._pi = pi
        self._support_n = support_n
        self._expansion_n = expansion_n

        self.as_matrices = as_matrices
        self.uniform = uniform

        self._support = np.linspace(0, pi, num=self._support_n + 1)[1:]
        self._densities = self.pdf(self._support)
        self._scores = self.score(self._support)
        self._cumulative_densities = np.cumsum(
            (self._support[1] - self._support[0]) * self._densities, axis=0
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"var={self._var:.3f}, "
            f"support_n={self._support_n}, "
            f"expansion_n={self._expansion_n})"
        )

    @property
    def var(self) -> float:
        """The variance of the distribution."""
        return self._var

    @property
    def support(self) -> np.ndarray:
        """The support of the distribution, i.e. the range of values over which densities are calculated."""
        return self._support

    @property
    def densities(self) -> np.ndarray:
        """The densities of the distribution over the support."""
        return self._densities

    @property
    def scores(self) -> np.ndarray:
        """
        The scalar-valued scores of the distribution over the support. To get the actual score - which
        is a 3-D vector in the tangent space of SO(3) - multiply the scalar values by the axis of rotation.
        """
        return self._scores

    @property
    def cumulative_densities(self) -> np.ndarray:
        """The cumulative density for each value in the support."""
        return self._cumulative_densities

    def inf_sum(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """Infinite sum in the IGSO3 distribution."""
        if isinstance(angle, float) or isinstance(angle, int):
            angle = np.array([angle], dtype=np.float64)

        expansion_steps = np.arange(self._expansion_n)[None, :]

        return np.sum(
            ((2 * expansion_steps) + 1)
            * np.exp(-expansion_steps * (expansion_steps + 1) * self._var)
            * (
                np.sin((expansion_steps + (1 / 2)) * angle[:, None])
                / np.sin(angle[:, None] / 2)
            ),
            axis=-1,
        )

    def pdf(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gives the probability density for some angle(s) under the parameterised IGSO3 distribution with some
        specified number of terms to expand the infinite series (see https://arxiv.org/pdf/2210.01776.pdf).
        """

        density = (1 - np.cos(angle)) / self._pi

        if not self.uniform:
            density *= self.inf_sum(angle)

        return density

    def cdf(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gives the cumulative density for some angle(s) under the parameterised IGSO3 distribution
        (see https://arxiv.org/pdf/2210.01776.pdf).
        """

        if isinstance(angle, float) or isinstance(angle, int):
            angle = np.array([angle], dtype=np.float64)

        densities = np.resize(
            self._densities[None, :], (angle.shape[0], self._densities.shape[0])
        )
        support = np.resize(
            self._support[None, :], (angle.shape[0], self._support.shape[0])
        )

        angle_support_index = np.argmin(np.abs(angle[:, None] - support), axis=-1)

        angle_support_values = support[np.arange(support.shape[0]), angle_support_index]
        zeroed_densities = np.where(
            support > angle_support_values[:, None], 0.0, densities
        )

        return np.trapz(zeroed_densities, x=support)

    def inv_cdf(self, cumulative_density: Union[float, np.ndarray]) -> np.ndarray:
        """
        Inverse of the cumulative density function, taking a cumulative density value as
        input and returning the correct value on the distribution's support.
        """
        assert np.all(cumulative_density >= 0), "The cumulative density must be >= 0"
        assert np.all(cumulative_density <= 1), "The cumulative density must be <1"

        return np.interp(cumulative_density, self._cumulative_densities, self._support)

    def score(self, angle: Union[float, np.ndarray], eps: float = 1e-12) -> np.ndarray:
        """
        Gets the gradient of the log PDF at a given angle or array of angles.
        Specifically this computes d log f(w)/dw via df(w)/dw * 1/f(w) (quotient rule).
        The argument `eps` is for numerical stability, and is added to the divisor.
        """
        if isinstance(angle, float) or isinstance(angle, int):
            angle = np.array([angle], dtype=np.float64)

        expansion_steps = np.arange(self._expansion_n)[None, :]
        a = expansion_steps + 0.5

        angle_expanded = angle[:, None]
        cos_half_angle = np.cos(angle_expanded / 2)
        cos_a_angle = np.cos(a * angle_expanded)
        sin_a_angle = np.sin(a * angle_expanded)
        sin_half_angle = np.sin(angle_expanded / 2)

        inf_sum_constant_terms = ((2 * expansion_steps) + 1) * np.exp(
            -expansion_steps * (expansion_steps + 1) * self._var
        )

        inf_sum = np.sum(
            inf_sum_constant_terms * (sin_a_angle / sin_half_angle),
            axis=-1,
        )
        inf_sum_derivative = np.sum(
            inf_sum_constant_terms
            * (
                ((a * cos_a_angle) / sin_half_angle)
                - ((cos_half_angle * sin_a_angle) / (2 * sin_half_angle**2))
            ),
            axis=-1,
        )

        return inf_sum_derivative / (inf_sum + eps)

    @staticmethod
    def sample_axis(size: SampleSize) -> np.ndarray:
        """
        Uniformly samples a random axis for rotation.

        Generates 3 variables from Gaussian (0,1), then normalizes. Method proven in
        Marsaglia, 1972. https://mathworld.wolfram.com/SpherePointPicking.html
        """

        if size == 1:
            size = tuple()
        elif isinstance(size, int):
            size = (size,)

        vec = np.random.normal(size=size + (3,))
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

        return vec

    def sample_angle(self, size: SampleSize) -> np.ndarray:
        """
        Samples a random angle for rotation according to Eq. 5 in https://openreview.net/forum?id=BY88eBbkpe5
        """

        if isinstance(size, tuple):
            cdfs = np.random.rand(*size)
        else:
            cdfs = np.random.rand(size)

        angle = self.inv_cdf(cdfs)

        return angle

    def sample(
        self, size: SampleSize, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Samples one or more rotation matrices according to `size`, returned as a tensor of shape `size + (3, 3)`
        or (3, 3) if `size` is 1.
        """

        axes = torch.as_tensor(
            self.sample_axis(size), dtype=torch.float32, device=device
        )
        angles = torch.as_tensor(
            self.sample_angle(size), dtype=torch.float32, device=device
        )

        if self.as_matrices:
            return axis_angle_to_matrix(axes, angles).squeeze(0)

        return axes * angles
