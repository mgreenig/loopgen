""" Tests the various distributions in loopgen.distributions."""

import numpy as np
import pytest
import torch
from math import pi
from loopgen.distributions import IGSO3Distribution, Gaussian3DDistribution
from loopgen.utils import is_positive_semidefinite, is_symmetric


class Gaussian3DDistributionTest:

    """
    Tests the Gaussian3DDistribution class.
    """

    mean = torch.tensor([0.0, 0.0, 0.0])
    variance = 1.0
    non_iso_cov_matrix = torch.ones(9).reshape(3, 3)
    non_iso_cov_matrix += torch.eye(3) * 2
    non_iso_cov_matrix += non_iso_cov_matrix.T.clone()

    distribution = Gaussian3DDistribution(mean, variance)
    non_iso_distribution = Gaussian3DDistribution(mean, non_iso_cov_matrix)

    def test_mean(self):
        assert torch.all(
            torch.eq(self.distribution.mean, self.mean)
        ).item(), "Inputted mean and distribution mean are not equal"

    def test_cov_matrix(self):
        """
        Check that the stored covariance matrix - when a scalar variance is entered -
        is positive semi-definite and symmetric.
        """

        for distr in [self.distribution, self.non_iso_distribution]:
            assert is_symmetric(distr.cov_matrix), "Covariance matrix is not symmetric"
            assert is_positive_semidefinite(
                distr.cov_matrix
            ), "Covariance matrix is not positive semi-definite"

            assert is_symmetric(distr.prec_matrix), "Precision matrix is not symmetric"
            assert is_positive_semidefinite(
                distr.prec_matrix
            ), "Precision matrix is not positive semidefinite"

            prec_matrix = torch.linalg.inv(distr.cov_matrix)
            assert torch.allclose(
                distr.prec_matrix, prec_matrix
            ), "Precision matrix is not the inverse of the covariance matrix"

    def test_pdf(self):
        sample = torch.as_tensor([0.1, 0.2, 0.3])
        pdf_value = self.distribution.pdf(sample)
        expected_pdf_value = torch.exp(self.distribution._distribution.log_prob(sample))
        assert np.allclose(
            pdf_value.item(), expected_pdf_value.item()
        ), "PDF values do not match true values"

    def test_is_isotropic(self):
        assert self.distribution.is_isotropic, "Distribution should be isotropic"
        assert (
            not self.non_iso_distribution.is_isotropic
        ), "Distribution should not be isotropic"


class TestIGSO3Distribution:

    """
    Tests the IGSO3 distribution.
    """

    default = IGSO3Distribution()

    def test_init(self):
        with pytest.raises(AssertionError):
            IGSO3Distribution(-1)
        with pytest.raises(AssertionError):
            IGSO3Distribution(0)
        with pytest.raises(AssertionError):
            IGSO3Distribution(support_n=-1)
        with pytest.raises(AssertionError):
            IGSO3Distribution(support_n=0)
        with pytest.raises(AssertionError):
            IGSO3Distribution(expansion_n=-1)
        with pytest.raises(AssertionError):
            IGSO3Distribution(expansion_n=0)

    def test_sample(self):
        """
        Rotation matrix must be valid. IGSO(3) has been visually confirmed
        as described at the top of the class
        """

        rot_mat = self.default.sample(1)

        assert rot_mat.shape == (
            3,
            3,
        ), "Sampled rotation matrix should have shape (3, 3)"

        det = np.linalg.det(rot_mat)
        assert np.isclose(
            det, 1.0, atol=1e-5
        ), "Sampled rotation matrix does not have a determinant of 1"

        inv = np.linalg.inv(rot_mat)
        transpose = np.transpose(rot_mat)
        assert np.all(
            np.isclose(inv, transpose, atol=1e-5)
        ), "Sampled rotation matrix is not orthogonal"

        # test sampling for different values of size
        sizes = [2, 20, (3, 4)]
        for size in sizes:
            rot_mat = self.default.sample(size)
            if isinstance(size, int):
                size = (size,)
            assert rot_mat.shape == size + (
                3,
                3,
            ), "Sampled rotation matrix should have shape size + (3, 3)"

    def test_pdf(self):
        pdf_default = self.default._densities
        assert np.allclose(
            pdf_default[0], 0.0, atol=1e-6
        ), "0 should have 0 probability"

        new = IGSO3Distribution(support_n=5)
        assert len(new._densities == 5), "pdf should only have size of support_n"

    def test_cdf(self):
        assert np.allclose(
            self.default.cdf(0), 0.0, atol=1e-3
        ), "cumulative density at 0º should be 0"
        assert np.allclose(
            self.default.cdf(pi), 1.0, atol=1e-3
        ), "cumulative density at pi should be 1"

    def test_inv_cdf(self):
        with pytest.raises(AssertionError):
            self.default.inv_cdf(-1)

        # self.assertAlmostEqual(default.inv_cdf(0),0, delta = 1e-6,  msg = "inv cdf of 0 should be 0")
        # self.assertAlmostEqual(default.inv_cdf(1e-9),0, delta = 1e-3, msg = "inv cdf of 1e-9 should be 0")
        assert np.allclose(
            self.default.inv_cdf(1), pi, atol=1e-3
        ), "inv cdf of 0 should be 0"

    def test_score(self):
        score = self.default.score(1.0)
        eps = 1e-12
        # estimate score with finite differences
        estimated_score = (
            np.log(self.default.inf_sum(1.0 + eps)) - np.log(self.default.inf_sum(1.0))
        ) / 1e-12
        assert np.allclose(
            score, estimated_score, atol=1e-2
        ), "Score estimated with finite differences should be within 0.01 of calculated score."

    def test_sample_axis(self):
        """
        checks axis is normalized
        """
        vec = self.default.sample_axis(1)
        norm = np.linalg.norm(vec)
        assert np.allclose(norm, 1.0), "Sampled axis norm is not 1"

    def test_sample_angle(self):
        """
        sample is combo of axis and angle, so if it works these are fine
        """
        angle = self.default.sample_angle(1)
        assert 0 <= angle <= pi, "Sampled angle is not in [0, pi]"
