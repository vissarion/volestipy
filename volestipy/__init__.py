"""
volestipy - Python bindings for the volesti C++ library.

This module re-exports the C++ pybind11 extension and adds
convenient Python-level wrappers and utilities.

Quick start
-----------
>>> import numpy as np
>>> from volestipy import HPolytope

# 3-D cube [-1,1]^3  (6 inequalities)
>>> d = 3
>>> A = np.vstack([np.eye(d), -np.eye(d)])
>>> b = np.ones(2 * d)
>>> P = HPolytope(A, b)
>>> samples = P.uniform_sample(n_samples=500, walk_type='cdhr')  # shape (3, 500)
>>> vol = P.volume(error=0.1)
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── locate the compiled extension ────────────────────────────────────────────
# When installed the .so sits inside the volestipy package directory.
# When built in-place (build_ext --inplace) it sits at the repo root.
def _import_extension():
    try:
        from volestipy import _volestipy  # type: ignore[no-redef]
        return _volestipy
    except ImportError:
        pass

    # Try the repo root (in-place build)
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    try:
        import _volestipy  # type: ignore[import]
        return _volestipy
    except ImportError as exc:
        raise ImportError(
            "Could not import the compiled _volestipy extension.\n"
            "Please build the extension first:\n"
            "  pip install -e .\n"
            "or:\n"
            "  mkdir build && cd build\n"
            "  cmake .. -DVOLESTI_INCLUDE_DIR=/path/to/volesti/include\n"
            "  cmake --build . -j4\n"
        ) from exc


_ext = _import_extension()

# ── Re-export the pybind11 classes ───────────────────────────────────────────
from volestipy._volestipy import (  # noqa: E402  # type: ignore[import]
    HPolytope as _HPolytope,
    hpoly_volume,
    hpoly_sample,
    VPolytope as _VPolytope,
    vpoly_volume,
    vpoly_sample,
)


# ── High-level Python wrappers ────────────────────────────────────────────────

class HPolytope:
    """
    H-Polytope: a convex polytope in half-space representation.

        P = { x ∈ R^d : A x ≤ b }

    Parameters
    ----------
    A : array-like of shape (m, d)
        Constraint matrix.
    b : array-like of shape (m,)
        Right-hand-side vector.

    Examples
    --------
    >>> import numpy as np
    >>> from volestipy import HPolytope
    >>> d = 3
    >>> A = np.vstack([np.eye(d), -np.eye(d)])
    >>> b = np.ones(2 * d)
    >>> P = HPolytope(A, b)
    >>> P.dimension()
    3
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        A = np.asarray(A, dtype=float, order="C")
        b = np.asarray(b, dtype=float).ravel()
        if A.ndim != 2:
            raise ValueError("A must be a 2-D array.")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("b must be a 1-D array with length A.shape[0].")
        self._poly = _HPolytope(A, b)

    # ── Metadata ─────────────────────────────────────────────────────────────
    def dimension(self) -> int:
        """Return the ambient dimension d."""
        return self._poly.dimension()

    def num_of_hyperplanes(self) -> int:
        """Return the number of half-space constraints."""
        return self._poly.num_of_hyperplanes()

    @property
    def A(self) -> np.ndarray:
        """Constraint matrix A (shape m×d)."""
        return np.array(self._poly.get_mat())

    @property
    def b(self) -> np.ndarray:
        """Right-hand-side vector b (shape m,)."""
        return np.array(self._poly.get_vec())

    def is_in(self, point: np.ndarray) -> bool:
        """
        Check if *point* is inside the polytope.

        Returns
        -------
        bool
        """
        p = np.asarray(point, dtype=float).ravel()
        return self._poly.is_in(p) == -1

    def compute_inner_ball(self):
        """
        Compute the largest inscribed (Chebyshev) ball.

        Returns
        -------
        center : numpy.ndarray of shape (d,)
        radius : float
        """
        c, r = self._poly.compute_inner_ball()
        return np.array(c), float(r)

    def normalize(self) -> "HPolytope":
        """Normalize each row of A to unit Euclidean norm (in-place)."""
        self._poly.normalize()
        return self

    # ── Sampling ─────────────────────────────────────────────────────────────
    def sample(
        self,
        n_samples: int = 1000,
        walk_length: int = 1,
        burn_in: int = 0,
        walk_type: str = "cdhr",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Draw uniform samples from the polytope.

        Parameters
        ----------
        n_samples : int
            Number of output points.
        walk_length : int
            Number of MCMC steps between consecutive samples (thinning).
        burn_in : int
            Number of warm-up steps discarded before sampling.
        walk_type : str
            MCMC walk type. Supported options:

            * ``'cdhr'`` - Coordinate Directions Hit-and-Run (default)
            * ``'rdhr'`` - Random Directions Hit-and-Run
            * ``'ball_walk'`` - Ball Walk
            * ``'billiard'`` - Billiard Walk (Accelerated Billiard Walk)
            * ``'dikin'`` - Dikin Walk
            * ``'john'`` - John Walk
            * ``'vaidya'`` - Vaidya Walk
        seed : int
            Random number generator seed.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
            Sample points stored column-wise.
        """
        return np.array(
            self._poly.uniform_sample(n_samples, walk_length, burn_in, walk_type, seed)
        )

    def gaussian_sample(
        self,
        n_samples: int = 1000,
        walk_length: int = 1,
        burn_in: int = 0,
        a: float = 1.0,
        walk_type: str = "cdhr",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Draw samples from the Gaussian distribution exp(-a ‖x‖²) restricted
        to this polytope.

        Parameters
        ----------
        n_samples, walk_length, burn_in, seed : see :meth:`sample`.
        a : float
            Variance parameter (larger → more concentrated).
        walk_type : str
            ``'cdhr'``, ``'rdhr'``, or ``'ball_walk'``.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        """
        return np.array(
            self._poly.gaussian_sample(n_samples, walk_length, burn_in, a, walk_type, seed)
        )

    def exponential_sample(
        self,
        c: np.ndarray,
        n_samples: int = 1000,
        walk_length: int = 1,
        burn_in: int = 0,
        a: float = 1.0,
        walk_type: str = "exponential_hmc",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Draw samples from the exponential distribution exp(a · cᵀ x) restricted
        to this polytope.

        Parameters
        ----------
        c : array-like of shape (d,)
            Bias direction.
        n_samples, walk_length, burn_in, seed : see :meth:`sample`.
        a : float
            Scaling parameter.
        walk_type : str
            ``'exponential_hmc'``.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        """
        c = np.asarray(c, dtype=float).ravel()
        return np.array(
            self._poly.exponential_sample(n_samples, walk_length, burn_in, c, a, walk_type, seed)
        )

    # ── Volume ────────────────────────────────────────────────────────────────
    def volume(
        self,
        error: float = 0.1,
        walk_length: int = 1,
        algorithm: str = "cooling_balls",
        walk_type: str = "cdhr",
    ) -> float:
        """
        Estimate the volume of the polytope.

        Parameters
        ----------
        error : float
            Relative error bound (e.g. 0.1 = 10 %).
        walk_length : int
            Thinning parameter.
        algorithm : str
            Volume algorithm:

            * ``'cooling_balls'``   (default, recommended)
            * ``'cooling_gaussians'``
            * ``'sequence_of_balls'``
        walk_type : str
            ``'cdhr'``, ``'rdhr'``, ``'ball_walk'``, ``'billiard'``.

        Returns
        -------
        float
            Estimated volume.
        """
        return float(self._poly.volume(error, walk_length, algorithm, walk_type))

    # ── Rounding ─────────────────────────────────────────────────────────────
    def round_min_ellipsoid(self):
        """
        Round the polytope by transforming the minimum enclosing ellipsoid
        of a sample to the unit ball.

        Returns
        -------
        T : numpy.ndarray of shape (d, d)
            Linear transformation matrix.
        T_shift : numpy.ndarray of shape (d,)
            Translation vector.
        round_val : float
            Rounding quality metric.
        """
        T, T_shift, rv = self._poly.round_min_ellipsoid()
        return np.array(T), np.array(T_shift), float(rv)

    def round_max_ellipsoid(self):
        """
        Round the polytope by transforming the maximum inscribed ellipsoid
        to the unit ball.

        Returns
        -------
        T : numpy.ndarray of shape (d, d)
        T_shift : numpy.ndarray of shape (d,)
        round_val : float
        """
        T, T_shift, rv = self._poly.round_max_ellipsoid()
        return np.array(T), np.array(T_shift), float(rv)

    # ── Dunder ────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"HPolytope(dimension={self.dimension()}, "
            f"num_hyperplanes={self.num_of_hyperplanes()})"
        )


class VPolytope:
    """
    V-Polytope: a convex polytope given as the convex hull of a finite set of
    vertices.

        P = conv{ v₁, v₂, …, vₙ }

    Parameters
    ----------
    V : array-like of shape (n_vertices, d)
        Matrix whose rows are the vertices.

    Examples
    --------
    >>> import numpy as np
    >>> from volestipy import VPolytope
    # 2-D triangle
    >>> V = np.array([[0., 0.], [1., 0.], [0., 1.]])
    >>> P = VPolytope(V)
    >>> P.dimension()
    2
    """

    def __init__(self, V: np.ndarray):
        V = np.asarray(V, dtype=float, order="C")
        if V.ndim != 2:
            raise ValueError("V must be a 2-D array with shape (n_vertices, d).")
        self._poly = _VPolytope(V)

    # ── Metadata ─────────────────────────────────────────────────────────────
    def dimension(self) -> int:
        """Return the ambient dimension d."""
        return self._poly.dimension()

    def num_of_vertices(self) -> int:
        """Return the number of vertices."""
        return self._poly.num_of_vertices()

    @property
    def V(self) -> np.ndarray:
        """Vertex matrix (shape n_vertices×d)."""
        return np.array(self._poly.get_mat())

    def is_in(self, point: np.ndarray) -> bool:
        """
        Check membership of *point* in the convex hull.

        Returns
        -------
        bool
        """
        p = np.asarray(point, dtype=float).ravel()
        return self._poly.is_in(p) == -1

    def compute_inner_ball(self):
        """
        Compute an inscribed ball.

        Returns
        -------
        center : numpy.ndarray of shape (d,)
        radius : float
        """
        c, r = self._poly.compute_inner_ball()
        return np.array(c), float(r)

    # ── Sampling ─────────────────────────────────────────────────────────────
    def sample(
        self,
        n_samples: int = 1000,
        walk_length: int = 1,
        burn_in: int = 0,
        walk_type: str = "cdhr",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Draw uniform samples from the V-polytope.

        Parameters
        ----------
        n_samples, walk_length, burn_in, walk_type, seed : see
            :meth:`HPolytope.sample`.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        """
        return np.array(
            self._poly.uniform_sample(n_samples, walk_length, burn_in, walk_type, seed)
        )

    # ── Volume ────────────────────────────────────────────────────────────────
    def volume(
        self,
        error: float = 0.1,
        walk_length: int = 1,
        algorithm: str = "cooling_balls",
        walk_type: str = "cdhr",
    ) -> float:
        """
        Estimate the volume of the V-polytope.

        Parameters
        ----------
        error : float
        walk_length : int
        algorithm : str
            ``'cooling_balls'`` or ``'sequence_of_balls'``.
        walk_type : str

        Returns
        -------
        float
        """
        return float(self._poly.volume(error, walk_length, algorithm, walk_type))

    def __repr__(self) -> str:
        return (
            f"VPolytope(dimension={self.dimension()}, "
            f"num_vertices={self.num_of_vertices()})"
        )


# ── Convenience constructors ─────────────────────────────────────────────────

def hypercube(d: int, r: float = 1.0) -> HPolytope:
    """
    Create the hypercube [-r, r]^d as an H-polytope.

    Parameters
    ----------
    d : int
        Dimension.
    r : float
        Half-side length (default 1.0).

    Returns
    -------
    HPolytope
    """
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.full(2 * d, r)
    return HPolytope(A, b)


def hypersimplex(d: int) -> HPolytope:
    """
    Create the standard d-dimensional simplex as an H-polytope.

        S_d = { x ∈ ℝ^d : x_i ≥ 0, Σ x_i ≤ 1 }

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    HPolytope
    """
    A = np.vstack([-np.eye(d), np.ones((1, d))])
    b = np.concatenate([np.zeros(d), [1.0]])
    return HPolytope(A, b)


def cross_polytope(d: int, r: float = 1.0) -> HPolytope:
    """
    Create the cross-polytope (hyperoctahedron) as an H-polytope.

        C_d = { x : |x_1| + … + |x_d| ≤ r }

    Parameters
    ----------
    d : int
    r : float
        Radius (default 1.0).

    Returns
    -------
    HPolytope
    """
    from itertools import product as iproduct
    signs = list(iproduct(*[[-1, 1]] * d))
    A = np.array(signs, dtype=float)
    b = np.full(len(signs), r)
    return HPolytope(A, b)


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "HPolytope",
    "VPolytope",
    "hypercube",
    "hypersimplex",
    "cross_polytope",
    "hpoly_volume",
    "hpoly_sample",
    "vpoly_volume",
    "vpoly_sample",
]

__version__ = "0.1.0"
