# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
example_hypercube.py
--------------------
Demonstrates uniform and Gaussian sampling from a hypercube using volestipy,
and estimates its volume with different algorithms.
"""
import numpy as np
import sys

try:
    from volestipy import HPolytope, hypercube, hypersimplex, VPolytope
except ImportError:
    print("ERROR: volestipy is not installed or the C++ extension is not built.")
    print("Please follow the build instructions in README.md.")
    sys.exit(1)


def main():
    print("=" * 60)
    print("volestipy - hypercube example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build the 4-D hypercube [-1, 1]^4
    # ------------------------------------------------------------------
    d = 40
    P = hypercube(d)
    print(f"\nPolytope: {P}")
    center, radius = P.compute_inner_ball()
    print(f"Inscribed ball: center={np.round(center, 4)}, radius={radius:.4f}")

    # ------------------------------------------------------------------
    # 2. Uniform sampling
    # ------------------------------------------------------------------
    print("\n--- Uniform sampling ---")
    for wt in ["cdhr", "rdhr", "ball_walk", "billiard", "dikin"]:
        samples = P.sample(n_samples=500, walk_type=wt, seed=42)
        print(f"  walk={wt:20s}  shape={samples.shape}  "
              f"mean≈{samples.mean(axis=1).round(3)}")

    # ------------------------------------------------------------------
    # 3. Gaussian sampling
    # ------------------------------------------------------------------
    print("\n--- Gaussian sampling (a=2.0) ---")
    gsamples = P.gaussian_sample(n_samples=500, a=2.0, walk_type="cdhr", seed=0)
    print(f"  shape={gsamples.shape}  std≈{gsamples.std(axis=1).round(3)}")

    # ------------------------------------------------------------------
    # 4. Volume estimation
    # ------------------------------------------------------------------
    print("\n--- Volume estimation (true = 2^4 = 16) ---")
    for algo in ["cooling_balls", "sequence_of_balls", "cooling_gaussians"]:
        wt = "cdhr" if algo != "cooling_gaussians" else "cdhr"
        vol = P.volume(error=0.15, algorithm=algo, walk_type=wt)
        print(f"  algorithm={algo:25s}  vol≈{vol:.3f}  (error {abs(vol-16)/16*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. 3-D standard simplex
    # ------------------------------------------------------------------
    print("\n--- 3-D standard simplex ---")
    S = hypersimplex(3)
    print(f"Polytope: {S}")
    samples_s = S.sample(n_samples=500, seed=0)
    print(f"Samples shape: {samples_s.shape}")
    print(f"All non-negative:  {(samples_s >= -1e-9).all()}")
    print(f"All sum <= 1:      {(samples_s.sum(axis=0) <= 1 + 1e-9).all()}")
    vol_s = S.volume(error=0.2)
    print(f"Volume estimate: {vol_s:.5f}  (true ≈ {1.0/6:.5f})")
    
    # ------------------------------------------------------------------
    # 6. V-Polytope: 2-D triangle
    # ------------------------------------------------------------------
    print("\n--- 2-D triangle (VPolytope) ---")
    V = np.array([[0., 0.], [1., 0.], [0., 1.]])
    T = VPolytope(V)
    print(f"Polytope: {T}")
    t_samples = T.sample(n_samples=300, seed=0)
    print(f"Samples shape: {t_samples.shape}")
    in_count = sum(T.is_in(t_samples[:, i]) for i in range(t_samples.shape[1]))
    print(f"Points inside: {in_count}/{t_samples.shape[1]}")
    vol_t = T.volume(error=0.3)
    print(f"Volume estimate: {vol_t:.4f}  (true = 0.5)")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
