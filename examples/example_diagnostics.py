# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
MCMC diagnostics: ESS and PSRF
===============================

After drawing samples from a polytope with an MCMC walk, two standard
diagnostics tell us how well the chain has mixed:

Effective Sample Size (ESS)
    The number of effectively *independent* samples, correcting for
    autocorrelation.  Computed per coordinate via the FFT-based
    initial monotone sequence estimator.  Higher is better.

    Rule of thumb: aim for ESS >= 200 per coordinate.

Potential Scale Reduction Factor (Rhat, PSRF)
    Measures whether the chain has converged by comparing the variance
    within the first and second halves of the chain.  Based on
    Gelman & Rubin (1992) for the univariate version and Brooks &
    Gelman (1998) for the multivariate version.

    Rule of thumb: Rhat < 1.1 indicates good convergence.

This example:
  1. Samples from three polytopes: hypercube, cross-polytope, Birkhoff B(3)
  2. Computes ESS and PSRF for each walk type
  3. Shows how ESS and PSRF change with sample size (convergence study)

Run with:
    python examples/example_diagnostics.py
"""

import numpy as np
from volestipy import (
    hypercube,
    cross_polytope,
    birkhoff_polytope,
    ess,
    univariate_psrf,
    multivariate_psrf,
)

SEP = "=" * 65


# --- helper ---
def diagnostics_row(samples):
    """Return (min_ess, max_univariate_psrf, multivariate_psrf) for a sample matrix."""
    ess_vals, min_e = ess(samples)
    u_psrf = univariate_psrf(samples)
    m_psrf = multivariate_psrf(samples)
    return int(min_e), float(u_psrf.max()), float(m_psrf)


def print_diagnostics_table(samples, walk_type, n_samples):
    min_e, max_upsrf, mpsrf = diagnostics_row(samples)
    print(
        f"  {walk_type:<15s}  n={n_samples:5d}  "
        f"min_ESS={min_e:5d}  "
        f"max_uPSRF={max_upsrf:.4f}  "
        f"mPSRF={mpsrf:.4f}"
    )


# --- 1. Walk-type comparison on the 10-D hypercube ---
print(SEP)
print("1. Walk-type comparison -- 10-D hypercube [-1,1]^10, n=2000")
print(SEP)

P_cube = hypercube(10)
walk_types = ["cdhr", "rdhr", "ball_walk", "billiard"]

for walk in walk_types:
    S = P_cube.sample(n_samples=2000, walk_type=walk, seed=42)
    print_diagnostics_table(S, walk, 2000)

# --- 2. Birkhoff polytope B(3) -- dimension 4 ---
print()
print(SEP)
print("2. Walk-type comparison -- Birkhoff B(3), dim=4, n=2000")
print(SEP)

P_bk = birkhoff_polytope(8)

for walk in walk_types:
    S = P_bk.sample(n_samples=2000, walk_type=walk, seed=42)
    print_diagnostics_table(S, walk, 2000)

# --- 3. ESS and PSRF per coordinate on the 5-D cross-polytope ---
print()
print(SEP)
print("3. Per-coordinate diagnostics -- 5-D cross-polytope, CDHR, n=3000")
print(SEP)

P_cross = cross_polytope(5)
S = P_cross.sample(n_samples=3000, walk_type="cdhr", seed=0)

ess_vals, min_e = ess(S)
u_psrf_vals = univariate_psrf(S)
m_psrf_val = multivariate_psrf(S)

print(f"  {'Coord':>5}  {'ESS':>8}  {'uPSRF':>8}  {'status'}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}")
for i in range(S.shape[0]):
    ess_ok  = "ok" if ess_vals[i] >= 200 else "fail"
    psrf_ok = "ok" if u_psrf_vals[i] < 1.1 else "fail"
    print(
        f"  {i:>5}  {ess_vals[i]:>8.1f}  {u_psrf_vals[i]:>8.4f}  "
        f"ESS{ess_ok} PSRF{psrf_ok}"
    )
print(f"\n  Multivariate PSRF: {m_psrf_val:.4f}  "
      f"({'converged ok' if m_psrf_val < 1.1 else 'not converged fail'})")

# --- 4. Convergence study: ESS vs sample size ---
print()
print(SEP)
print("4. Convergence study -- 5-D hypercube, CDHR, varying n")
print(SEP)
print(f"  {'n_samples':>10}  {'min_ESS':>8}  {'ESS/n':>8}  {'max_uPSRF':>10}  {'mPSRF':>8}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

P5 = hypercube(5)
for n in [200, 500, 1000, 2000, 5000]:
    S = P5.sample(n_samples=n, walk_type="cdhr", seed=7)
    min_e, max_upsrf, mpsrf = diagnostics_row(S)
    ess_ratio = min_e / n
    print(
        f"  {n:>10}  {min_e:>8}  {ess_ratio:>8.3f}  "
        f"{max_upsrf:>10.4f}  {mpsrf:>8.4f}"
    )

print()
print("Done.")
