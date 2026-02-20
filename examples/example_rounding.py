# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

"""
Rounding comparison: barrier methods vs. no rounding
=====================================================

Rounding transforms a polytope P → L⁻¹(P − shift) so that the result
is more "ball-like" (low condition number).  Good rounding dramatically
improves MCMC mixing and volume-estimation accuracy.

volestipy exposes five rounding methods:

  round_min_ellipsoid      — min covering ellipsoid of random samples
  round_max_ellipsoid      — max inscribed ellipsoid (John ellipsoid)
  round_log_barrier        — analytic center (log-barrier Hessian)
  round_volumetric_barrier — volumetric center
  round_vaidya_barrier     — Vaidya center (interpolates log ↔ volumetric)

Test polytope
-------------
Anisotropic box [-sᵢ, sᵢ]^d with half-widths sᵢ log-spaced in [1, 100].
Condition number ≈ 100.  True volume = ∏ 2sᵢ (known exactly).

The billiard walk is very sensitive to anisotropy: without rounding,
min_ESS ~ 3 on n=3000 samples; after rounding, min_ESS > 1000.

Volume accuracy
---------------
After rounding P → P' via transformation T, the original volume is
    vol(P) = vol(P') × round_val,  where round_val = |det(T)|.

Run with:
    python examples/example_rounding.py
"""

from __future__ import annotations

import numpy as np
from volestipy import HPolytope, ess, univariate_psrf, multivariate_psrf

# ─── polytope factory ────────────────────────────────────────────────────────

def anisotropic_box(d: int, cond: float = 100.0) -> HPolytope:
    """
    Axis-aligned box [-sᵢ, sᵢ]^d, half-widths sᵢ = logspace(1, cond, d).
    True volume = ∏ 2sᵢ.
    """
    scales = np.logspace(0, np.log10(cond), d)
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.concatenate([scales, scales])
    return HPolytope(A, b)


def true_volume(d: int, cond: float) -> float:
    scales = np.logspace(0, np.log10(cond), d)
    return float(np.prod(2 * scales))


# ─── helpers ─────────────────────────────────────────────────────────────────

def run_diagnostics(P: HPolytope, n: int, walk: str, seed: int = 0) -> dict:
    S = P.sample(n_samples=n, walk_type=walk, seed=seed)
    ess_vals, min_e = ess(S)
    u_psrf = univariate_psrf(S)
    m_psrf = multivariate_psrf(S)
    return dict(
        samples=S,
        min_ess=int(min_e),
        mean_ess=float(ess_vals.mean()),
        ess_vals=ess_vals,
        max_upsrf=float(u_psrf.max()),
        mpsrf=float(m_psrf),
    )


SEP = "=" * 72

# ─── parameters ──────────────────────────────────────────────────────────────

D = 8
COND = 100.0
N_SAMPLES = 3000
VOL_ERR = 0.1
TRUE_VOL = true_volume(D, COND)

ROUNDING_METHODS = [
    ("none",               None),
    ("min_ellipsoid",      "round_min_ellipsoid"),
    ("max_ellipsoid",      "round_max_ellipsoid"),
    ("log_barrier",        "round_log_barrier"),
    ("volumetric_barrier", "round_volumetric_barrier"),
    ("vaidya_barrier",     "round_vaidya_barrier"),
]

print(SEP)
print(f"Anisotropic box: d={D},  condition number ≈ {COND:.0f}")
print(f"Half-widths sᵢ = logspace(1, {COND:.0f}, {D})  →  true vol = {TRUE_VOL:.6g}")
print(SEP)

# ─── 1. Sampling diagnostics: CDHR and billiard walk ─────────────────────────

for walk in ["cdhr", "billiard"]:
    print()
    print(f"Walk: {walk}  —  n={N_SAMPLES} samples")
    print(f"  {'Method':<22} {'min_ESS':>8} {'mean_ESS':>9} {'max_uR̂':>8} {'mR̂':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")

    for label, method_name in ROUNDING_METHODS:
        P = anisotropic_box(D, COND)
        if method_name is not None:
            getattr(P, method_name)()
        d = run_diagnostics(P, n=N_SAMPLES, walk=walk, seed=0)
        print(
            f"  {label:<22} {d['min_ess']:>8d} {d['mean_ess']:>9.1f} "
            f"{d['max_upsrf']:>8.4f} {d['mpsrf']:>8.4f}"
        )

# ─── 2. Volume accuracy ───────────────────────────────────────────────────────

print()
print(SEP)
print(f"Volume estimation  (true vol = {TRUE_VOL:.6g})")
print(f"  vol(P) = vol(rounded P') × round_val,   round_val = |det(T)|")
print()
print(f"  {'Method':<22} {'round_val':>12} {'vol_est':>14} {'error %':>9}")
print(f"  {'-'*22} {'-'*12} {'-'*14} {'-'*9}")

for label, method_name in ROUNDING_METHODS:
    P = anisotropic_box(D, COND)
    round_val = None

    if method_name is not None:
        T, shift, round_val = getattr(P, method_name)()

    vol_rounded = P.volume(error=VOL_ERR, algorithm="cooling_balls")
    vol_est = vol_rounded * round_val if round_val else vol_rounded
    err_pct = abs(vol_est - TRUE_VOL) / TRUE_VOL * 100.0
    rv_str = f"{round_val:.4g}" if round_val is not None else "—"

    print(
        f"  {label:<22} {rv_str:>12} {vol_est:>14.6g} {err_pct:>8.1f}%"
    )

# ─── 3. Per-coordinate ESS: no rounding vs. Vaidya (billiard walk) ───────────

print()
print(SEP)
print("Per-coordinate ESS: no rounding vs. Vaidya barrier  (billiard, n=3000)")
print(f"  {'Coord':>5}  {'ESS (none)':>11}  {'ESS (vaidya)':>13}  {'gain':>6}")
print(f"  {'-'*5}  {'-'*11}  {'-'*13}  {'-'*6}")

P_none = anisotropic_box(D, COND)
d_none = run_diagnostics(P_none, n=N_SAMPLES, walk="billiard", seed=0)

P_vaidya = anisotropic_box(D, COND)
P_vaidya.round_vaidya_barrier()
d_vaidya = run_diagnostics(P_vaidya, n=N_SAMPLES, walk="billiard", seed=0)

for i in range(D):
    e_none = d_none["ess_vals"][i]
    e_vaidya = d_vaidya["ess_vals"][i]
    gain = e_vaidya / e_none if e_none > 0 else float("nan")
    print(f"  {i:>5}  {e_none:>11.1f}  {e_vaidya:>13.1f}  {gain:>5.1f}×")

print()
print("Done.")
