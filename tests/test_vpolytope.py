"""
Tests for volestipy.VPolytope - construction, membership, sampling, volume.
"""
import math
import numpy as np
import pytest

try:
    from volestipy import VPolytope
    HAVE_EXT = True
except ImportError:
    HAVE_EXT = False

pytestmark = pytest.mark.skipif(not HAVE_EXT,
    reason="C++ extension _volestipy not built")


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_simplex_vpoly(d: int) -> VPolytope:
    """Standard simplex as V-polytope: origin + d unit vectors."""
    V = np.vstack([np.zeros(d), np.eye(d)])
    return VPolytope(V)


def make_cube_vpoly_2d() -> VPolytope:
    """2-D square [-1,1]^2 as V-polytope."""
    V = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    return VPolytope(V)


# ── Construction tests ────────────────────────────────────────────────────────

class TestVPolytopeConstruction:
    def test_dimension(self):
        P = make_simplex_vpoly(3)
        assert P.dimension() == 3

    def test_num_vertices(self):
        d = 3
        P = make_simplex_vpoly(d)
        assert P.num_of_vertices() == d + 1  # d+1 vertices for simplex

    def test_get_V(self):
        V = np.array([[0., 0.], [1., 0.], [0., 1.]])
        P = VPolytope(V)
        np.testing.assert_array_almost_equal(P.V, V)

    def test_invalid_V(self):
        with pytest.raises((ValueError, RuntimeError)):
            VPolytope(np.array([1.0, 2.0]))  # 1-D input

    def test_repr(self):
        P = make_simplex_vpoly(3)
        assert "VPolytope" in repr(P)


# ── Membership tests ──────────────────────────────────────────────────────────

class TestVPolytopeMembership:
    def test_origin_in_simplex(self):
        P = make_simplex_vpoly(3)
        assert P.is_in(np.zeros(3))

    def test_centroid_in_simplex(self):
        d = 3
        P = make_simplex_vpoly(d)
        centroid = np.ones(d) / (d + 1)
        assert P.is_in(centroid)

    def test_vertex_in_simplex(self):
        P = make_simplex_vpoly(3)
        assert P.is_in(np.array([1., 0., 0.]))

    def test_outside_simplex(self):
        P = make_simplex_vpoly(3)
        # Point outside the simplex
        assert not P.is_in(np.array([2., 0., 0.]))

    def test_center_in_square(self):
        P = make_cube_vpoly_2d()
        assert P.is_in(np.zeros(2))

    def test_vertex_in_square(self):
        P = make_cube_vpoly_2d()
        assert P.is_in(np.array([1.0, 1.0]))


# ── Inner ball tests ──────────────────────────────────────────────────────────

class TestVPolytopeInnerBall:
    def test_inner_ball_square(self):
        P = make_cube_vpoly_2d()
        center, radius = P.compute_inner_ball()
        assert center.shape == (2,)
        assert radius > 0

    def test_inner_ball_simplex(self):
        P = make_simplex_vpoly(3)
        center, radius = P.compute_inner_ball()
        assert radius > 0
        assert P.is_in(center)


# ── Sampling tests ────────────────────────────────────────────────────────────

class TestVPolytopeUniformSampling:
    @pytest.mark.parametrize("walk_type", ["cdhr", "rdhr", "ball_walk", "billiard"])
    def test_sample_shape(self, walk_type):
        P = make_simplex_vpoly(3)
        samples = P.sample(n_samples=100, walk_type=walk_type, seed=42)
        assert samples.shape == (3, 100)

    def test_samples_inside_simplex(self):
        d = 3
        P = make_simplex_vpoly(d)
        samples = P.sample(n_samples=100, seed=0)
        for i in range(samples.shape[1]):
            assert P.is_in(samples[:, i]), f"Point {i} outside simplex"

    def test_sample_reproducible(self):
        P = make_simplex_vpoly(3)
        s1 = P.sample(n_samples=50, seed=5)
        s2 = P.sample(n_samples=50, seed=5)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_sample_inside_square(self):
        P = make_cube_vpoly_2d()
        samples = P.sample(n_samples=100, seed=0)
        assert samples.shape == (2, 100)
        for i in range(samples.shape[1]):
            assert P.is_in(samples[:, i])


# ── Volume tests ──────────────────────────────────────────────────────────────

class TestVPolytopeVolume:
    #cooling_gaussians not applicable to V-polytopes
    @pytest.mark.parametrize("volume_algo", ["cooling_balls", "sequence_of_balls"]) 
    def test_simplex_volume_2d(self, volume_algo):
        # Vol(triangle with vertices (0,0),(1,0),(0,1)) = 0.5
        V = np.array([[0., 0.], [1., 0.], [0., 1.]])
        P = VPolytope(V)
        vol = P.volume(error=0.3, algorithm=volume_algo)
        expected = 0.5
        assert abs(vol - expected) / expected < 1.0, f"vol={vol:.4f}"

    @pytest.mark.parametrize("volume_algo", ["cooling_balls", "sequence_of_balls"])
    def test_square_volume(self, volume_algo):
        P = make_cube_vpoly_2d()
        vol = P.volume(error=0.3, algorithm=volume_algo)
        expected = 4.0  # 2x2 square
        assert abs(vol - expected) / expected < 0.5, f"vol={vol:.3f}"

    @pytest.mark.parametrize("volume_algo", ["cooling_balls", "sequence_of_balls"])
    def test_volume_positive(self, volume_algo):
        P = make_simplex_vpoly(3)
        vol = P.volume(error=0.3, algorithm=volume_algo)
        assert vol > 0
