# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
Tests for volestipy.HPolytope - construction, membership, sampling, volume.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

# Skip all tests if the C++ extension is not built
try:
    from volestipy import HPolytope, hypercube, hypersimplex, cross_polytope
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False

pytestmark = pytest.mark.skipif(not HAVE_EXT,
    reason="C++ extension _volestipy not built")


# --- Helpers ---
def make_cube_hpoly(d: int, r: float = 1.0) -> HPolytope:
    """[-r, r]^d as an H-polytope."""
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.full(2 * d, r)
    return HPolytope(A, b)


def make_simplex_hpoly(d: int) -> HPolytope:
    """Standard simplex S_d = {x >= 0, sum(x) <= 1}."""
    A = np.vstack([-np.eye(d), np.ones((1, d))])
    b = np.concatenate([np.zeros(d), [1.0]])
    return HPolytope(A, b)


# --- Construction tests ---
class TestHPolytopeConstruction:
    def test_dimension(self):
        P = make_cube_hpoly(3)
        assert P.dimension() == 3

    def test_num_hyperplanes(self):
        d = 4
        P = make_cube_hpoly(d)
        assert P.num_of_hyperplanes() == 2 * d

    def test_get_A_b(self):
        d = 2
        A = np.vstack([np.eye(d), -np.eye(d)])
        b = np.ones(2 * d)
        P = HPolytope(A, b)
        np.testing.assert_array_almost_equal(P.A, A)
        np.testing.assert_array_almost_equal(P.b, b)

    def test_invalid_A_b(self):
        with pytest.raises((ValueError, RuntimeError)):
            HPolytope(np.eye(3), np.ones(2))  # shape mismatch

    def test_repr(self):
        P = make_cube_hpoly(3)
        assert "HPolytope" in repr(P)

    def test_convenience_hypercube(self):
        P = hypercube(3, r=2.0)
        assert P.dimension() == 3
        assert P.is_in(np.array([1.5, -1.5, 0.5]))
        assert not P.is_in(np.array([2.5, 0.0, 0.0]))

    def test_convenience_simplex(self):
        P = hypersimplex(3)
        assert P.dimension() == 3
        assert P.is_in(np.array([0.1, 0.2, 0.3]))
        assert not P.is_in(np.array([0.5, 0.5, 0.5]))  # sum > 1

    def test_convenience_cross_polytope(self):
        P = cross_polytope(3, r=1.0)
        assert P.dimension() == 3
        assert P.is_in(np.array([0.3, 0.3, 0.3]))  # sum = 0.9 <= 1


# --- Membership tests ---
class TestHPolytopeMembership:
    def test_origin_in_cube(self):
        P = make_cube_hpoly(5)
        assert P.is_in(np.zeros(5))

    def test_vertex_in_cube(self):
        P = make_cube_hpoly(3)
        assert P.is_in(np.ones(3))
        assert P.is_in(-np.ones(3))

    def test_outside_cube(self):
        P = make_cube_hpoly(3)
        assert not P.is_in(np.array([2.0, 0.0, 0.0]))

    def test_origin_in_simplex(self):
        P = make_simplex_hpoly(4)
        assert P.is_in(np.zeros(4))

    def test_outside_simplex(self):
        P = make_simplex_hpoly(3)
        assert not P.is_in(np.array([0.5, 0.5, 0.5]))  # sum = 1.5 > 1


# --- Inner ball tests ---
class TestHPolytopeInnerBall:
    def test_inner_ball_cube(self):
        P = make_cube_hpoly(3)
        center, radius = P.compute_inner_ball()
        assert center.shape == (3,)
        assert radius > 0
        # cube [-1,1]^3: Chebyshev ball has radius 1, center at origin
        np.testing.assert_array_almost_equal(center, np.zeros(3), decimal=5)
        assert abs(radius - 1.0) < 0.05

    def test_inner_ball_simplex(self):
        P = make_simplex_hpoly(2)
        center, radius = P.compute_inner_ball()
        assert radius > 0
        assert P.is_in(center)


# --- Sampling tests ---
class TestHPolytopeUniformSampling:
    @pytest.mark.parametrize("walk_type", [
        "cdhr", "rdhr", "ball_walk", "billiard"])
    def test_sample_shape(self, walk_type):
        P = make_cube_hpoly(4)
        samples = P.sample(n_samples=100, walk_type=walk_type, seed=42)
        assert samples.shape == (4, 100)

    def test_samples_inside_polytope(self):
        d = 3
        P = make_cube_hpoly(d)
        samples = P.sample(n_samples=200, seed=0)
        for i in range(samples.shape[1]):
            assert P.is_in(samples[:, i]), f"Point {i} outside polytope"

    def test_sample_reproducible(self):
        P = make_cube_hpoly(3)
        s1 = P.sample(n_samples=50, seed=7)
        s2 = P.sample(n_samples=50, seed=7)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_sample_different_seeds(self):
        P = make_cube_hpoly(3)
        s1 = P.sample(n_samples=50, seed=1)
        s2 = P.sample(n_samples=50, seed=2)
        assert not np.allclose(s1, s2)

    def test_sample_burn_in(self):
        P = make_cube_hpoly(3)
        samples = P.sample(n_samples=100, burn_in=50, seed=0)
        assert samples.shape == (3, 100)

    def test_sample_inside_simplex(self):
        d = 3
        P = make_simplex_hpoly(d)
        samples = P.sample(n_samples=100, seed=0)
        for i in range(samples.shape[1]):
            assert P.is_in(samples[:, i]), f"Sample {i} outside simplex"

    def test_invalid_walk_type(self):
        P = make_cube_hpoly(3)
        with pytest.raises((RuntimeError, ValueError, Exception)):
            P.sample(walk_type="unknown_walk_xyz")

    def test_billiard(self):
        P = make_cube_hpoly(4)
        samples = P.sample(n_samples=50, walk_type="billiard", seed=0)
        assert samples.shape == (4, 50)

    @pytest.mark.parametrize("walk_type", ["dikin", "john", "vaidya"])
    def test_barrier_walks(self, walk_type):
        P = make_cube_hpoly(4)
        samples = P.sample(n_samples=50, walk_type=walk_type, seed=0)
        assert samples.shape == (4, 50)


class TestHPolytopeGaussianSampling:
    def test_gaussian_sample_shape(self):
        P = make_cube_hpoly(3)
        samples = P.gaussian_sample(n_samples=100, a=1.0, seed=0)
        assert samples.shape == (3, 100)

    def test_gaussian_sample_inside(self):
        d = 3
        P = make_cube_hpoly(d)
        samples = P.gaussian_sample(n_samples=100, seed=0)
        for i in range(samples.shape[1]):
            assert P.is_in(samples[:, i])

    @pytest.mark.parametrize("walk_type", ["cdhr", "rdhr", "ball_walk"])
    def test_gaussian_walk_types(self, walk_type):
        P = make_cube_hpoly(3)
        samples = P.gaussian_sample(n_samples=50, walk_type=walk_type, seed=0)
        assert samples.shape == (3, 50)


# --- Volume tests ---
class TestHPolytopeVolume:
    """
    Volume of the d-cube [-1,1]^d = 2^d.
    We use loose tolerances because these are Monte Carlo estimators.
    """

    @pytest.mark.parametrize("algorithm", [
        "cooling_balls", "sequence_of_balls", "cooling_gaussians"])
    def test_cube_volume_2d(self, algorithm):
        d = 2
        P = make_cube_hpoly(d)
        walk_type = "cdhr" if algorithm != "cooling_gaussians" else "cdhr"
        vol = P.volume(error=0.2, algorithm=algorithm, walk_type=walk_type)
        expected = 2.0 ** d  # = 4
        assert abs(vol - expected) / expected < 0.5, \
            f"algorithm={algorithm}: vol={vol:.3f}, expected={expected}"

    def test_cube_volume_3d_cooling_balls(self):
        d = 3
        P = make_cube_hpoly(d)
        vol = P.volume(error=0.2, algorithm="cooling_balls", walk_type="cdhr")
        expected = 2.0 ** d  # = 8
        assert abs(vol - expected) / expected < 0.5, f"vol={vol:.3f}, expected={expected}"

    def test_simplex_volume_3d(self):
        # Vol(S_3) = 1/6
        d = 3
        P = make_simplex_hpoly(d)
        vol = P.volume(error=0.3, algorithm="cooling_balls", walk_type="cdhr")
        expected = 1.0 / math.factorial(d)
        assert abs(vol - expected) / expected < 1.0, \
            f"vol={vol:.5f}, expected={expected:.5f}"

    def test_volume_positive(self):
        P = make_cube_hpoly(4)
        vol = P.volume()
        assert vol > 0

    def test_invalid_algorithm(self):
        P = make_cube_hpoly(3)
        with pytest.raises((RuntimeError, ValueError, Exception)):
            P.volume(algorithm="nonexistent_algo")
