"""
Code for diffusion model, including forward and reverse processes.
"""

from typing import Tuple, Literal

from abc import ABC, abstractmethod
import os

import torch
import pickle
import numpy as np
import logging as lg
from torch import nn
from torch_geometric.typing import PairTensor

from ..utils import so3_exp_map, so3_log_map
from ..distributions import IGSO3Distribution, Gaussian3DDistribution


BetaScheduleTypes = Literal["linear", "cosine", "quadratic", "logarithmic", "sigmoid"]


# Variance schedules
def cosine_beta_schedule(
    num_time_steps: int,
    min_beta: float,
    max_beta: float,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672.
    """
    steps = num_time_steps + 1
    x = torch.linspace(0, num_time_steps, steps)
    alphas_cumprod = (
        torch.cos(((x / num_time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min_beta, max_beta)


def linear_beta_schedule(
    num_time_steps: int,
    min_beta: float,
    max_beta: float,
) -> torch.Tensor:
    """Linearly-increasing sequence of betas."""
    return torch.linspace(min_beta, max_beta, num_time_steps)


def quadratic_beta_schedule(
    num_time_steps: int,
    min_beta: float,
    max_beta: float,
) -> torch.Tensor:
    """Quadratically-increasing sequence of betas."""
    return torch.linspace(min_beta**0.5, max_beta**0.5, num_time_steps) ** 2


def logarithmic_beta_schedule(
    num_time_steps: int,
    min_beta: torch.tensor,
    max_beta: torch.tensor,
):
    """Logarithmically-increasing sequence of betas."""
    t = torch.linspace(0, 1, num_time_steps)
    return torch.log(
        t * torch.exp(torch.tensor(max_beta))
        + (1 - t) * torch.exp(torch.tensor(min_beta))
    )


def sigmoid_beta_schedule(
    num_time_steps: int,
    min_var: float,
    max_var: float,
    min_beta: int = -6,
    max_beta: int = 6,
) -> torch.Tensor:
    """Sigmoid-scaled sequence of variances."""
    betas = torch.linspace(min_beta, max_beta, num_time_steps)
    return torch.sigmoid(betas) * (max_var - min_var) + min_var


def cumulative_integral(
    values: torch.tensor, delta_t: torch.float = 1.0
) -> torch.Tensor:
    """
    Returns cumulative integral of values, which should be the  at every value from 0 to T
    in a rank-1 tensor with the same length as `values`.
    """
    values = values.unsqueeze(0).expand(values.shape[0], -1) * (delta_t / len(values))
    return torch.trapezoid(torch.tril(values), dim=-1)


def betas_to_variances(
    betas: torch.tensor, delta_t: torch.float = 1.0, cumulative: bool = True
):
    """
    Converts from a variance schedule of betas to actual variances for the corresponding
    time step distributions.

    Returns the exponential term from Eqn. 29 of https://arxiv.org/pdf/2011.13456.pdf for VP-SDE
    Note that the variance is 1 - exponential, and that the mean at time t is
    sqrt(exponential) * x_0.

    If `cumulative` is True, this returns a sequence of variances corresponding to
    `Var[ q(x_t | x_0) ]`. Otherwise, this returns a sequence of variances corresponding
    to `Var[ q(x_{t+1} | x_t) ]`.
    """
    if not cumulative:
        dt = delta_t / len(betas)
        betas_t_1 = torch.nn.functional.pad(betas, (1, 0))[:-1]
        integrals = (betas + betas_t_1) * dt / 2
    else:
        integrals = cumulative_integral(betas, delta_t)

    return 1 - torch.exp(-integrals)


class BetaScheduleSelector:
    """
    Returns a beta schedule when called. Note we use the term `beta` here differently
    than some resources in the DDPM literature. Specifically `beta(t)` is the diffusion
    coefficient of our forward process stochastic differential equation, and so
    the actual variances of the forward process are determined by some function of
    `beta(t)` but are not necessarily `beta(t)` itself.
    """

    def __init__(
        self,
        num_time_steps: int,
        min_beta: float,
        max_beta: float,
    ):
        self._num_time_steps = num_time_steps
        self._min_beta = min_beta
        self._max_beta = max_beta

    def __call__(self, schedule_type: BetaScheduleTypes) -> torch.Tensor:
        """Gets a schedule according the input string."""

        if schedule_type == "linear":
            schedule = linear_beta_schedule(
                self._num_time_steps, self._min_beta, self._max_beta
            )
        elif schedule_type == "logarithmic":
            schedule = logarithmic_beta_schedule(
                self._num_time_steps, self._min_beta, self._max_beta
            )
        elif schedule_type == "cosine":
            schedule = cosine_beta_schedule(
                self._num_time_steps, self._min_beta, self._max_beta
            )
        elif schedule_type == "quadratic":
            schedule = quadratic_beta_schedule(
                self._num_time_steps, self._min_beta, self._max_beta
            )
        elif schedule_type == "sigmoid":
            schedule = sigmoid_beta_schedule(
                self._num_time_steps, self._min_beta, self._max_beta
            )
        else:
            raise ValueError(
                f"Invalid schedule type {schedule_type}. Should be one of {BetaScheduleTypes.__args__}"
            )

        return schedule


class DiffusionForwardProcess(ABC):

    """
    Abstract class for the forward process in a diffusion model,
    which samples scores (gradients of the log density) for a
    set of input means and a provided time step using
    the corresponding forward process distribution.

    This class takes a single argument - the variance schedule beta(t) -
    which is a tensor of shape (num_time_steps,). Note that the betas are
    not the actual variances of any distribution. The variances are computed
    from the betas according to:

        **Var[ p(x_t | x_t') ] = 1 - exp(-integral(beta, t, t'))**

    Which is just the solution to the Fokker-planck equation.
    """

    def __init__(self, betas: torch.Tensor):
        self._betas = betas
        # from the betas, get the stepwise variances i.e. Var[q(x_{t+1}| x_t)]
        self._stepwise_vars = betas_to_variances(betas, cumulative=False)

        # from the betas, get the conditional variances i.e. Var[q(x_t| x_0)]
        self._conditional_vars = betas_to_variances(betas)

    @abstractmethod
    def sample(self, inputs: torch.Tensor, time_step: int) -> PairTensor:
        """
        Generates samples and scores from the distribution for the input
        time step, conditioned on the corresponding inputs.
        """
        pass


class IGSO3ForwardProcess(DiffusionForwardProcess):

    """
    Diffusion model over the manifold of rotations SO(3).
    Samples rotations using the IGSO(3) distribution.
    """

    def __init__(
        self,
        betas: torch.Tensor,
        support_n: int = 2000,
        expansion_n: int = 2000,
        use_cache: bool = True,
        cache_path: str = "igso3_cache.pkl",
    ):
        """
        :param betas: The beta values to use to calculate variances of the diffusion process.
            Beta is related to the variances by the following equation:
            Var(q_t | q_t') = 1 - exp(-integral(beta, t', t)) -
            for any two times t' and t.
        :param support_n: The number of support points to use for the IGSO3 distributions.
        :param expansion_n: The number of terms to use in the approximation of the infinite
            series in the IGSO3 distributions.
        :param use_cache: Whether to use a cache of precomputed values for the IGSO3 distribution.
            If True, the cache will be loaded from the path specified by cache_path if it exists,
            and calculated and saved to that path if the cache does not exist. If False, the cache
            will be ignored and the IGSO3 distributions will be computed from scratch.
        :param cache_path: The path to the cache of precomputed values for the IGSO3 distribution.
        """

        super().__init__(betas)

        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                igso3_dists = pickle.load(f)
                (
                    self._stepwise_distributions,
                    self._conditional_distributions,
                ) = igso3_dists

                if not len(self._stepwise_distributions) == len(
                    self._conditional_distributions
                ):
                    raise ValueError(
                        "Lengths of cached stepwise and conditional distributions do not match."
                    )

                for i in range(len(self._stepwise_distributions)):
                    stepwise_dist = self._stepwise_distributions[i]
                    cond_dist = self._conditional_distributions[i]
                    if stepwise_dist.var != self._stepwise_vars[i]:
                        lg.warning(
                            "Variances for one or more cached stepwise "
                            "distributions do not match the input beta schedule."
                        )
                    if cond_dist.var != self._conditional_vars[i]:
                        lg.warning(
                            "Variances for one or more cached conditional "
                            "distributions do not match the input beta schedule."
                        )

        else:
            # distributions for sampling q(x_{t+1} | x_t) for each time step
            self._stepwise_distributions = [
                IGSO3Distribution(
                    v.item(), support_n=support_n, expansion_n=expansion_n
                )
                for v in self._stepwise_vars
            ]

            # distributions for sampling q(x_t | x_0) for each time step
            self._conditional_distributions = [
                IGSO3Distribution(
                    v.item(), support_n=support_n, expansion_n=expansion_n
                )
                for v in self._conditional_vars
            ]

            if use_cache:
                pickle.dump(
                    (self._stepwise_distributions, self._conditional_distributions),
                    open(cache_path, "wb"),
                )

    @staticmethod
    def rotation_matrix_angle(R: torch.Tensor, eps=1e-6):
        """Gets the angle of rotation around the axis of rotation of a rotation matrix."""
        # multiplying by (1-epsilon) prevents instability of arccos when provided with -1 or 1 as input.
        R_ = R.to(torch.float64)
        trace = torch.diagonal(R_, dim1=-2, dim2=-1).sum(dim=-1) * (1 - eps)
        out = (trace - 1.0) / 2.0
        out = torch.clamp(out, min=-0.99, max=0.99)
        return torch.arccos(out).to(R.dtype)

    def sample(
        self,
        inputs: torch.Tensor,
        time_step: int,
        from_x0: bool = True,
    ) -> PairTensor:
        """
        Samples rotations and the corresponding score matrices
        according to the corresponding time step distribution.

        If `from_x0` is True, the new rotations are sampled by
        treating the inputs as x_0 and sampling from q(x_t | x_0).
        Otherwise, the new rotations are sampled by treating the inputs
        as x_t and sampling from q(x_{t+1} | x_t).
        """

        if from_x0:
            distribution = self._conditional_distributions[time_step]
        else:
            distribution = self._stepwise_distributions[time_step]

        if len(inputs.shape) > 2:
            size = inputs.shape[0]
        else:
            size = 1

        sampled_matrix = distribution.sample(size=size, device=inputs.device)
        sampled_matrix = torch.matmul(sampled_matrix, inputs)
        rotation_diff = torch.matmul(inputs.transpose(-1, -2), sampled_matrix)

        scores = self.get_scores(rotation_diff, distribution)

        return sampled_matrix, scores

    def get_scores(
        self, samples: torch.Tensor, distribution: IGSO3Distribution
    ) -> torch.Tensor:
        """
        Get the score(s) (i.e. gradient of the log density) of the
        corresponding distribution for the sampled rotations.

        Argument `samples` should be a [..., 3, 3] tensor of rotation matrices.

        This returns a tensor of 3-vectors of shape [..., 3] representing
        the scores of the input rotations represented as euclidean vectors
        in the tangent space at the identity. These can be converted to true
        Lie algebra elements (skew-symmetric matrices) via loopgen.utils.so3_hat().
        """

        sampled_angles = self.rotation_matrix_angle(samples)

        angle_scores = torch.as_tensor(
            np.interp(
                sampled_angles.detach().cpu().numpy(),
                distribution.support,
                distribution.scores,
            ),
            device=sampled_angles.device,
            dtype=samples.dtype,
        )
        score_vector = so3_log_map(samples) / sampled_angles.unsqueeze(-1)
        rotation_scores = score_vector * angle_scores.unsqueeze(-1)

        return rotation_scores


class Gaussian3DForwardProcess(DiffusionForwardProcess):

    """
    Forward process in three dimensions using a gaussian distribution.
    This assumes the input gaussian distributions have identical variance
    across components and zero covariance.
    """

    def __init__(self, betas: torch.Tensor):
        super().__init__(betas)
        self._beta_integrals = cumulative_integral(self._betas)

        self._stepwise_distributions = []
        self._conditional_distributions = []
        for i in range(len(self._betas)):
            # distribution for sampling q(x_{t+1} | x_t) for each time step
            stepwise_distr = Gaussian3DDistribution(
                torch.zeros(3), self._stepwise_vars[i].item()
            )
            self._stepwise_distributions.append(stepwise_distr)

            # distribution for sampling q(x_t | x_0) for each time step
            cond_distr = Gaussian3DDistribution(
                torch.zeros(3), self._conditional_vars[i].item()
            )
            # print(i, self._conditional_vars[i], flush = True)
            self._conditional_distributions.append(cond_distr)

    def sample(
        self,
        inputs: torch.Tensor,
        time_step: int,
        from_x0: bool = True,
        normalise_score: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a 3-D vector and its score from the corresponding Gaussian distribution.

        :param inputs: The input vectors that are the means of the noise distribution.
        :param time_step: The time step to sample from.
        :param from_x0: If True, the new vectors are sampled by treating
            the inputs as x_0 and sampling from q(x_t | x_0). Otherwise,
            the new rotations are sampled by treating the inputs
            as x_t and sampling from q(x_{t+1} | x_t).
        :param normalise_score: If True, the score is normalised by multiplying by the
            -1 * standard deviation of the distribution, meaning that the score
            itself is distributed as a standard normal.
        :returns: A 2-tuple of the sampled vector and the normalised score.
        """

        if from_x0:
            distribution = self._conditional_distributions[time_step]
            marginal_b_t = self._beta_integrals[time_step]
            adjusted_means = torch.exp(-0.5 * marginal_b_t) * inputs
        else:
            distribution = self._stepwise_distributions[time_step]
            adjusted_means = torch.sqrt(1 - self._stepwise_vars[time_step]) * inputs

        if len(inputs.shape) > 1:
            size = inputs.shape[0]
        else:
            size = 1

        distribution_sample = distribution.sample(size=size, device=inputs.device)

        samples = adjusted_means + distribution_sample
        score = self.get_scores(adjusted_means, samples, distribution)

        if normalise_score:
            if from_x0:
                std = torch.sqrt(self._conditional_vars[time_step])
            else:
                std = torch.sqrt(self._stepwise_vars[time_step])
            score *= -std

        return samples, score

    @staticmethod
    def get_scores(
        means: torch.Tensor,
        samples: torch.Tensor,
        distribution: Gaussian3DDistribution,
    ) -> torch.Tensor:
        """
        Gets the 3-D score vectors for some inputs and samples generated
        from a given distribution at a given time step.
        """

        # score computation can be simplified for isotropic variance
        if distribution.is_isotropic:
            var = distribution.cov_matrix[0, 0]
            scores = (means - samples) / var
        else:
            scores = (
                -torch.matmul(
                    distribution.prec_matrix
                    + distribution.prec_matrix.transpose(-2, -1),
                    (samples - distribution.mean),
                )
                / 2
            )

        return scores


class SE3ForwardProcess:
    """
    Diffusion forward process for the group SE(3), i.e. the group of all affine
    transformations (rotations/translations).

    Uses the SO3ForwardProcess class to diffuse rotations towards the uniform IGSO(3)
    distribution and the Gaussian3DForwardProcess class to diffuse translations
    towards the standard normal.
    """

    def __init__(
        self,
        rotation_betas: torch.Tensor,
        translation_betas: torch.Tensor,
        igso3_support_n: int = 2000,
        igso3_expansion_n: int = 2000,
    ):
        """
        Initialises the class by creating the forward diffusion process classes
        from variance schedules.
        """

        rotation_forward_process = IGSO3ForwardProcess(
            rotation_betas,
            support_n=igso3_support_n,
            expansion_n=igso3_expansion_n,
        )
        translation_forward_process = Gaussian3DForwardProcess(translation_betas)

        self._rotation_forward_process = rotation_forward_process
        self._translation_forward_process = translation_forward_process

    def sample(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        time_step: int,
        from_x0: bool = True,
    ) -> Tuple[PairTensor, PairTensor, torch.Tensor]:
        """
        Generates samples and corresponding scores for the rotation and translation forward processes.
        """
        (
            sampled_rotations,
            sampled_rotation_scores,
        ) = self._rotation_forward_process.sample(rotations, time_step, from_x0=from_x0)
        (
            sampled_translations,
            sampled_translation_scores,
            sampled_translation_noise,
        ) = self._translation_forward_process.sample(
            translations, time_step, from_x0=from_x0
        )

        return (
            (sampled_rotations, sampled_translations),
            (
                sampled_rotation_scores,
                sampled_translation_scores,
            ),
            sampled_translation_noise,
        )


class DiffusionReverseProcess(nn.Module, ABC):
    """
    Abstract class for the reverse process in a diffusion model,
    which uses an input score and generates samples using the
    Euler-Maruyama method for the SDE Solver.
    """

    def __init__(
        self,
        beta_schedule: torch.Tensor,
        score_dim: int = 3,
        noise_scale: float = 0.0,
    ):
        super().__init__()

        self.register_buffer("_standard_normal_mean", torch.zeros(score_dim))
        self.register_buffer("_standard_normal_cov", torch.eye(score_dim))
        self.register_buffer(
            "_var_schedule", torch.zeros(beta_schedule.shape)
        )  # placeholder to be compatible with earlier checkpoint
        self.register_buffer("_noise_scale", torch.as_tensor(noise_scale))
        self._dt = torch.tensor(1.0 / len(beta_schedule))
        self._standard_normal = Gaussian3DDistribution(
            self._standard_normal_mean, self._standard_normal_cov
        )

    @abstractmethod
    def forward(self, inputs: torch.Tensor, score: torch.Tensor, time_step: int):
        """
        Samples from the reverse process, using the score and incrementing `inputs`
        with an Euler-Maruyama step using `score` and some brownian motion.
        """

        pass


class SO3ReverseProcess(DiffusionReverseProcess):

    """
    This class implements the reverse process on SO3 as a geodesic
    random walk on the manifold of 3D rotations as described
    in De Bortoli et al. (2022) (https://arxiv.org/abs/2202.02763).

    **Note that the variance schedule passed to this class should be the conditional variances -
    i.e. Var[p(x_t | x_0)] - rather than the stepwise variances - i.e. Var[p(x_t+1 | x_t)]**.
    """

    def __init__(
        self,
        beta_schedule: torch.Tensor,
        score_dim: int = 3,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            beta_schedule=beta_schedule, score_dim=score_dim, noise_scale=noise_scale
        )

        self._var_schedule = cumulative_integral(beta_schedule)

    def get_diffusion_coef(self, time_step: int) -> torch.Tensor:
        """
        Returns the diffusion coefficient g(t) for the Langevin dynamics, using the parameterization introduced
        in FrameDiff on page 34, where g_r(s) = sqrt(d/ds sigma^2(s)). Numerical differentiation as approximation...
        """
        return torch.sqrt(
            (self._var_schedule[time_step] - self._var_schedule[time_step - 1])
            / self._dt
        )

    def forward(
        self,
        current_rotations: torch.Tensor,
        score: torch.Tensor,
        time_step: int,
    ) -> torch.Tensor:
        """
        Increments current rotation matrices using the predicted score and the SO(3)
        exponential map to get an updated rotation for each input rotation.

        :param current_rotations: The current rotation matrices (R_t) of shape (..., 3, 3).
        :param score: The score for each rotation matrix (s_t). This is represented
            as a tensor of 3-d euclidean vectors, of shape (..., 3).
        :param time_step: The current time step.
        :returns: Batch of updated rotation matrices (R_t-1), of shape (..., 3, 3).
        """

        standard_gaussian_sample = self._standard_normal.sample(
            score.shape[0], device=score.device
        )
        # get a random tangent element to the manifold at the identity rotation
        tangent_gaussian = standard_gaussian_sample

        # Euler-Maruyama step in the identity's tangent space using the score and step size
        g_t = self.get_diffusion_coef(time_step)

        # The 2 is not in FrameDiff, however, we found that this reverse step with the analytically computed noise
        # only correctly moved backwards when the 2 was applied. Empirically this led to higher quality samples.
        so3_update = (
            g_t**2 * score * self._dt * 2
            + g_t * torch.sqrt(self._dt) * self._noise_scale * tangent_gaussian
        )

        # convert the tangent element to an actual rotation matrix
        update_rotation = so3_exp_map(so3_update)

        # get the new rotation
        updated_rotations = torch.matmul(current_rotations, update_rotation)

        return updated_rotations


class R3ReverseProcess(DiffusionReverseProcess):

    """
    Here we implement the reverse process on R3 using the DDPM formulation from
    Ho et al https://arxiv.org/pdf/2006.11239.pdf. Note that this is the same method used by FrameDiff,
    as they show that this is equivalent to SGBM with a smart choice of lambda.
    """

    def __init__(
        self,
        beta_schedule: torch.Tensor,
        score_dim: int = 3,
        noise_scale: float = 0.0,
    ):
        super().__init__(
            beta_schedule=beta_schedule, score_dim=score_dim, noise_scale=noise_scale
        )

        self._stepwise_var_schedule = betas_to_variances(
            beta_schedule, cumulative=False
        )
        self._conditional_var_schedule = betas_to_variances(
            beta_schedule, cumulative=True
        )

    def forward(
        self,
        current_translations: torch.Tensor,
        score: torch.Tensor,
        time_step: int,
    ) -> torch.Tensor:
        """
        Forward pass to update a set of translations with a tensor of scores.
        Note that under the current formulation, `score` should be normalised so it is N(0,I) distributed
        at timestep t for each residue. This is equivalent to the true score * -sigma, where sigma
        is the standard deviation of the noise at timestep t.
        """

        # Euler-Maruyama step using the score and step size
        standard_gaussian_sample = self._standard_normal.sample(
            score.shape[0], device=score.device
        )

        beta_t = self._stepwise_var_schedule[time_step]
        one_minus_alpha_bar_t = self._conditional_var_schedule[time_step]

        updated_coords = (
            1
            / torch.sqrt(1 - beta_t)
            * (
                current_translations
                - beta_t / torch.sqrt(one_minus_alpha_bar_t) * score
            )
        )

        noise = (
            self._noise_scale * torch.sqrt(beta_t) * standard_gaussian_sample
        )  # From Ho et al., assuming sigma(t)^2 = Beta(t)

        updated_translations = updated_coords + noise

        return updated_translations
