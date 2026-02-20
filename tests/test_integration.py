# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
Integration tests: compare sampling statistics against analytical ground truth
and verify volume estimates are within acceptable Monte Carlo bounds.
"""
import math
import numpy as np
import pytest

try:
    from volestipy import HPolytope, VPolytope, hypercube, hypersimplex
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False

try:
    from volestipy._volestipy import vpoly_sample as _vpoly_sample  # noqa: F401
    HAVE_VPOLY = True
except (ImportError, AttributeError):
    HAVE_VPOLY = False

pytestmark = pytest.mark.skipif(not HAVE_EXT,
    reason="C++ extension _volestipy not built")


def cube_volume(d, r=1.0):
    return (2 * r) ** d


def simplex_volume(d):
    return 1.0 / math.factorial(d)


# --- Module-level free-function tests ---
class TestFreeFunctions:
    def test_hpoly_sample_function(self):
        from volestipy._volestipy import hpoly_sample
        d = 3
        A = np.vstack([np.eye(d), -np.eye(d)])
        b = np.ones(2 * d)
        samples = hpoly_sample(A, b, n_samples=100, seed=0)
        assert samples.shape == (d, 100)

    def test_hpoly_volume_function(self):
        from volestipy._volestipy import hpoly_volume
        d = 2
        A = np.vstack([np.eye(d), -np.eye(d)])
        b = np.ones(2 * d)
        vol = hpoly_volume(A, b, error=0.2, algorithm="cooling_balls")
        expected = cube_volume(d)
        assert abs(vol - expected) / expected < 0.5

    @pytest.mark.skipif(not HAVE_VPOLY, reason="VPolytope not available (DISABLE_LPSOLVE)")
    def test_vpoly_sample_function(self):
        from volestipy._volestipy import vpoly_sample
        V = np.array([[0., 0.], [1., 0.], [0., 1.]])
        samples = vpoly_sample(V, n_samples=100, seed=0)
        assert samples.shape == (2, 100)

    @pytest.mark.skipif(not HAVE_VPOLY, reason="VPolytope not available (DISABLE_LPSOLVE)")
    def test_vpoly_volume_function(self):
        from volestipy._volestipy import vpoly_volume
        V = np.array([[0., 0.], [1., 0.], [0., 1.]])
        vol = vpoly_volume(V, error=0.3, algorithm="cooling_balls")
        assert vol > 0


# --- Statistical sanity checks ---
class TestSamplingStatistics:
    """Verify that samples have roughly the correct mean and covariance."""

    def test_cube_sample_mean(self):
        """Uniform samples from [-1,1]^d should have zero mean."""
        d = 4
        P = hypercube(d)
        samples = P.sample(n_samples=2000, burn_in=200, seed=42)
        mean = samples.mean(axis=1)
        np.testing.assert_allclose(mean, np.zeros(d), atol=0.15,
                                   err_msg="Sample mean of hypercube should be ~0")

    def test_cube_sample_std(self):
        """Std dev of Uniform[-1,1] is 1/sqrt(3) ~= 0.577."""
        d = 4
        P = hypercube(d)
        samples = P.sample(n_samples=2000, burn_in=200, seed=0)
        std = samples.std(axis=1)
        expected_std = 1.0 / math.sqrt(3)
        np.testing.assert_allclose(std, np.full(d, expected_std), atol=0.1)

    def test_simplex_sample_mean(self):
        """Uniform samples from standard simplex S_d should have mean 1/(d+1)."""
        d = 3
        P = hypersimplex(d)
        samples = P.sample(n_samples=2000, burn_in=200, seed=1)
        mean = samples.mean(axis=1)
        expected = np.full(d, 1.0 / (d + 1))
        np.testing.assert_allclose(mean, expected, atol=0.1)

    def test_gaussian_sample_concentration(self):
        """Gaussian samples with large a should be more concentrated near 0."""
        d = 3
        P = hypercube(d)
        s_low = P.gaussian_sample(n_samples=500, a=0.01, seed=0)
        s_high = P.gaussian_sample(n_samples=500, a=100.0, seed=0)
        # High a -> more concentrated -> smaller std
        assert s_high.std() < s_low.std() + 0.3


# --- High-dimensional smoke tests ---
class TestHighDimensional:
    """Basic sanity checks for higher dimensions."""

    @pytest.mark.parametrize("d", [10, 15])
    def test_cube_sampling_high_dim(self, d):
        P = hypercube(d)
        samples = P.sample(n_samples=200, walk_length=10, seed=0)
        assert samples.shape == (d, 200)
        # all coords should be in [-1, 1]
        assert (np.abs(samples) <= 1.0 + 1e-9).all()

    def test_volume_high_dim(self):
        d = 6
        P = hypercube(d)
        vol = P.volume(error=0.3, algorithm="cooling_balls")
        expected = cube_volume(d)
        assert vol > 0
        # loose bound: within 2x of true value
        assert vol < 2 * expected
