# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
Birkhoff polytope example
=========================

The Birkhoff polytope B(n) is the convex polytope of nxn doubly stochastic
matrices -- non-negative real matrices whose rows and columns all sum to 1.

By the Birkhoff-von Neumann theorem, the vertices of B(n) are exactly the
nxn permutation matrices, so B(n) has n! vertices.

This example demonstrates:
  1. Creating B(n) for several values of n
  2. Uniform sampling (CDHR walk)
  3. Recovering doubly stochastic matrices from samples
  4. Volume approximation

Run with:
    python examples/example_birkhoff.py
"""

import numpy as np
from volestipy import birkhoff_polytope


# --- helper: reconstruct the full nxn doubly stochastic matrix ---
def reduced_to_full(x: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct the full nxn doubly stochastic matrix from a reduced
    coordinate vector x of length d = (n-1)^2.

    The first (n-1)^2 entries of the doubly stochastic matrix (row-major,
    excluding the last row and last column) are exactly the coordinates used
    by generate_birkhoff.  The last row and column are recovered by the
    sum-to-one constraints.

    Parameters
    ----------
    x : ndarray, shape ((n-1)^2,)
        A point in B(n) as returned by volestipy.
    n : int
        Matrix size.

    Returns
    -------
    M : ndarray, shape (n, n)
        Full doubly stochastic matrix.
    """
    M = np.zeros((n, n))
    # Fill the top-left (n-1)x(n-1) block
    M[: n - 1, : n - 1] = x.reshape(n - 1, n - 1)
    # Last column: row sums must equal 1
    M[: n - 1, n - 1] = 1.0 - M[: n - 1, : n - 1].sum(axis=1)
    # Last row: column sums must equal 1
    M[n - 1, :] = 1.0 - M[: n - 1, :].sum(axis=0)
    return M


def check_doubly_stochastic(M: np.ndarray, tol: float = 1e-9) -> bool:
    """Return True if M is doubly stochastic up to tolerance."""
    n = M.shape[0]
    rows_ok = np.allclose(M.sum(axis=1), np.ones(n), atol=tol)
    cols_ok = np.allclose(M.sum(axis=0), np.ones(n), atol=tol)
    nonneg = (M >= -tol).all()
    return bool(rows_ok and cols_ok and nonneg)


# --- 1. Basic properties ---
print("=" * 60)
print("Birkhoff polytope B(n) -- basic properties")
print("=" * 60)

for n in range(2, 6):
    P = birkhoff_polytope(n)
    expected_dim = (n - 1) ** 2
    expected_facets = n * n
    print(
        f"  B({n}):  dim = {P.dimension():3d}  (expected {expected_dim:3d}),  "
        f"facets = {P.num_of_hyperplanes():3d}  (expected {expected_facets:3d})"
    )

# --- 2. Uniform sampling from B(3) ---
print()
print("=" * 60)
print("Uniform sampling from B(3)  (CDHR walk, 200 samples)")
print("=" * 60)

n = 3
P3 = birkhoff_polytope(n)
samples = P3.sample(n_samples=200, walk_type="cdhr", seed=42)
# samples shape: (d, n_samples) = (4, 200)

print(f"  Sample matrix shape: {samples.shape}  (d x n_samples)")

# Verify each sample gives a valid doubly stochastic matrix
bad = 0
for i in range(samples.shape[1]):
    M = reduced_to_full(samples[:, i], n)
    if not check_doubly_stochastic(M):
        bad += 1

print(f"  Doubly stochastic check: {samples.shape[1] - bad}/{samples.shape[1]} samples pass")

# Show one example
x0 = samples[:, 0]
M0 = reduced_to_full(x0, n)
print(f"\n  First sample -- reduced coordinates: {np.round(x0, 4)}")
print("  Reconstructed 3x3 doubly stochastic matrix:")
print("  " + str(np.round(M0, 4)).replace("\n", "\n  "))
print(f"  Row sums: {np.round(M0.sum(axis=1), 6)}")
print(f"  Col sums: {np.round(M0.sum(axis=0), 6)}")

# --- 3. Volume approximation ---
print()
print("=" * 60)
print("Volume approximation (cooling balls, error=0.1)")
print("=" * 60)

# Known exact volumes for reference:
#   vol(B(2)) = 1           (line segment [0,1])
#   vol(B(3)) = 3/4         (4-dim polytope, exact = 0.09375 in std coords,
#                             but volesti uses a specific basis so differs)
for n in [2, 3, 4, 5, 6]:
    P = birkhoff_polytope(n)
    vol = P.volume(error=0.01, algorithm="cooling_balls")
    print(f"  vol(B({n})) = {vol:.6g}   (dim={P.dimension()})")

# --- 4. Multiple walk types on B(3) ---
print()
print("=" * 60)
print("Walk-type comparison on B(3)  (500 samples each)")
print("=" * 60)

walk_types = ["cdhr", "rdhr", "ball_walk", "billiard"]
for walk in walk_types:
    S = P3.sample(n_samples=500, walk_type=walk, seed=0)
    mean_pt = S.mean(axis=1)
    print(f"  {walk:15s}  sample mean ~= {np.round(mean_pt, 3)}")

print()
print("Done.")
