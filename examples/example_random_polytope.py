# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
example_random_polytope.py
--------------------------
Builds a random full-dimensional polytope by sampling rows of A from N(0,I),
then estimates its volume and draws uniform samples.
"""
import numpy as np
import sys

try:
    from volestipy import HPolytope
except ImportError:
    print("volestipy not installed. See README.md for build instructions.")
    sys.exit(1)


def random_hpoly(d: int, m: int, seed: int = 0) -> HPolytope:
    """
    Create a random H-polytope in dimension d with m constraints.

    Each constraint a_i^T x <= 1 where a_i ~ N(0, I_d).
    Intersected with the unit ball: ||x|| <= R for safety.
    """
    rng = np.random.default_rng(seed)
    A_rand = rng.standard_normal((m, d))
    b_rand = np.ones(m)
    # Add bounding box to make it bounded
    A = np.vstack([A_rand, np.eye(d), -np.eye(d)])
    b = np.concatenate([b_rand, 2 * np.ones(d), 2 * np.ones(d)])
    return HPolytope(A, b)


def main():
    print("=" * 60)
    print("volestipy - random polytope example")
    print("=" * 60)

    d = 5    # dimension
    m = 20   # number of random constraints

    P = random_hpoly(d, m, seed=42)
    print(f"\nRandom polytope: dim={P.dimension()}, constraints={P.num_of_hyperplanes()}")

    center, radius = P.compute_inner_ball()
    print(f"Inscribed ball radius: {radius:.4f}")
    print(f"Inscribed ball center: {np.round(center, 3)}")

    # Uniform sampling
    print("\n--- Sampling ---")
    samples = P.sample(n_samples=1000, walk_length=5, burn_in=200,
                       walk_type="billiard", seed=0)
    print(f"Samples shape: {samples.shape}")
    print(f"Sample mean:   {np.round(samples.mean(axis=1), 3)}")
    print(f"Sample std:    {np.round(samples.std(axis=1), 3)}")

    # Check all samples inside
    inside = sum(P.is_in(samples[:, i]) for i in range(samples.shape[1]))
    print(f"Fraction inside: {inside}/{samples.shape[1]}")

    # Volume
    print("\n--- Volume ---")
    vol = P.volume(error=0.2, algorithm="cooling_balls", walk_type="cdhr")
    print(f"Estimated volume: {vol:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
