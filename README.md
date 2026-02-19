# volestipy

Python bindings for the [volesti](https://github.com/GeomScale/volesti) C++ library —
a high-performance library for **volume approximation** and **sampling** of convex bodies
(H-polytopes, V-polytopes).

[![CI](https://github.com/vissarion/volestipy/actions/workflows/ci.yml/badge.svg)](https://github.com/vissarion/volestipy/actions/workflows/ci.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

---

## Features

| Feature | Details |
|---|---|
| **Polytope types** | `HPolytope` (`Ax ≤ b`) · `VPolytope` (convex hull of vertices) |
| **Uniform sampling** | CDHR · RDHR · Ball Walk · Billiard Walk · Accelerated Billiard Walk · Dikin Walk · John Walk · Vaidya Walk |
| **Gaussian sampling** | Gaussian CDHR · Gaussian RDHR · Gaussian Ball Walk |
| **Exponential sampling** | Exponential HMC |
| **Volume algorithms** | Cooling Balls · Cooling Gaussians · Sequence of Balls |
| **Rounding** | Minimum covering ellipsoid · Maximum inscribed ellipsoid |
| **Convenience constructors** | `hypercube(d)` · `hypersimplex(d)` · `cross_polytope(d)` |

---

## Requirements

| Dependency | Version |
|---|---|
| C++ compiler | GCC ≥ 9 or Clang ≥ 11, C++17 |
| CMake | ≥ 3.15 |
| Python | ≥ 3.8 |
| pybind11 | ≥ 2.11 |
| Eigen3 | ≥ 3.3 |
| Boost | ≥ 1.56 (headers + `random`, `math`) |

Optional:
* **LP-solve 5.5** - needed for VPolytope inner ball computation (otherwise auto-disabled)

---

## Installation

### 1. Clone with the volesti submodule

```bash
git clone https://github.com/GeomScale/volestipy.git
cd volestipy
git submodule update --init --recursive   # clones volesti into external/volesti
```

If volesti is already installed elsewhere, skip submodule init and set
`VOLESTI_INCLUDE_DIR` (see below).

### 2. Install system dependencies (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake build-essential \
    libeigen3-dev \
    libboost-all-dev \
    python3-dev python3-pip
```

### 3. Install Python dependencies

```bash
pip install pybind11 numpy
```

### 4a. Build via `pip` (recommended)

```bash
# With volesti as a submodule (default):
pip install .

# With a custom volesti path:
VOLESTI_INCLUDE_DIR=/path/to/volesti/include pip install .
```

### 4b. Manual CMake build (in-place)

```bash
mkdir build && cd build
cmake .. \
    -DVOLESTI_INCLUDE_DIR=../external/volesti/include \
    -DCMAKE_BUILD_TYPE=Release \
    -DDISABLE_LPSOLVE=ON
cmake --build . -j$(nproc)
cd ..
# Copy/symlink the .so into the package directory or add build/ to PYTHONPATH
```

### Verify

```python
import volestipy
print(volestipy.__version__)   # 0.1.0
```

---

## Quick Start

```python
import numpy as np
from volestipy import HPolytope, VPolytope, hypercube, hypersimplex

# ── H-Polytope: 3-D hypercube [-1, 1]^3 ─────────────────────────────────────
d = 3
P = hypercube(d)

# Uniform sampling (1 000 points, CDHR walk)
samples = P.sample(n_samples=1000, walk_type="cdhr", seed=42)
print(samples.shape)   # (3, 1000)

# Gaussian sampling
g_samples = P.gaussian_sample(n_samples=500, a=1.5, seed=0)

# Volume estimation (true = 8.0)
vol = P.volume(error=0.1, algorithm="cooling_balls")
print(f"Volume ≈ {vol:.2f}  (true = {2**d})")

# ── Build an arbitrary H-polytope ─────────────────────────────────────────────
A = np.vstack([np.eye(d), -np.eye(d)])   # 2d inequalities
b = 2.0 * np.ones(2 * d)
P2 = HPolytope(A, b)
center, radius = P2.compute_inner_ball()

# ── V-Polytope: 2-D triangle ─────────────────────────────────────────────────
V = np.array([[0., 0.], [1., 0.], [0., 1.]])
T = VPolytope(V)
t_samples = T.sample(n_samples=300, seed=0)
vol_t = T.volume(error=0.2)
```

---

## API Reference

### `HPolytope(A, b)`

```
P = { x ∈ ℝ^d : A x ≤ b }
```

| Method | Description |
|---|---|
| `dimension()` | Ambient dimension *d* |
| `num_of_hyperplanes()` | Number of constraints *m* |
| `A`, `b` | Constraint matrix / vector (properties) |
| `is_in(point)` | Membership test |
| `compute_inner_ball()` | Returns `(center, radius)` of inscribed ball |
| `normalize()` | Scale rows of *A* to unit norm (in-place) |
| `sample(n_samples, walk_length, burn_in, walk_type, seed)` | Uniform samples |
| `gaussian_sample(n_samples, ..., a, walk_type, seed)` | Gaussian samples |
| `exponential_sample(c, n_samples, ..., a, seed)` | Exponential samples |
| `volume(error, walk_length, algorithm, walk_type)` | Volume estimate |
| `round_min_ellipsoid()` | Round via min enclosing ellipsoid |
| `round_max_ellipsoid()` | Round via max inscribed ellipsoid |

### `VPolytope(V)`

```
P = conv{ rows of V }
```

| Method | Description |
|---|---|
| `dimension()` | Ambient dimension |
| `num_of_vertices()` | Number of vertices |
| `V` | Vertex matrix (property) |
| `is_in(point)` | Membership test |
| `compute_inner_ball()` | Returns `(center, radius)` |
| `sample(...)` | Uniform samples |
| `volume(...)` | Volume estimate |

### Walk types

| Name | Class | Polytopes |
|---|---|---|
| `"cdhr"` | Coordinate Directions Hit-and-Run | H, V |
| `"rdhr"` | Random Directions Hit-and-Run | H, V |
| `"ball_walk"` | Ball Walk | H, V |
| `"billiard"` | Billiard Walk | H, V |
| `"accelerated_billiard"` | Accelerated Billiard Walk | H |
| `"dikin"` | Dikin Walk | H |
| `"john"` | John Walk | H |
| `"vaidya"` | Vaidya Walk | H |
| `"exponential_hmc"` | Exponential HMC | H (exponential) |

### Volume algorithms

| Name | Notes |
|---|---|
| `"cooling_balls"` | Recommended, works for H & V |
| `"cooling_gaussians"` | H-polytope only |
| `"sequence_of_balls"` | H & V, faster but less accurate |

### Convenience constructors

```python
from volestipy import hypercube, hypersimplex, cross_polytope

P = hypercube(d, r=1.0)        # [-r, r]^d
S = hypersimplex(d)             # { x≥0, Σx_i ≤ 1 }
C = cross_polytope(d, r=1.0)   # { Σ|x_i| ≤ r }
```

---

## Examples

```bash
python examples/example_hypercube.py
python examples/example_random_polytope.py
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
volestipy/
├── CMakeLists.txt               # CMake build script
├── setup.py                     # Python packaging (CMake-based)
├── pyproject.toml
├── README.md
├── external/
│   └── volesti/                 # git submodule - volesti library
├── src/
│   └── bindings/
│       └── volesti_bindings.cpp # pybind11 binding definitions
├── volestipy/
│   └── __init__.py              # Python wrapper + convenience API
├── tests/
│   ├── conftest.py
│   ├── test_hpolytope.py
│   ├── test_vpolytope.py
│   └── test_integration.py
└── examples/
    ├── example_hypercube.py
    └── example_random_polytope.py
```

---

## Relationship to Other Projects

* **[volesti](https://github.com/GeomScale/volesti)** - the underlying C++ library
* **[dingo](https://github.com/GeomScale/dingo)** - metabolic network analysis using volesti
* **volestipy** - a standalone, general-purpose Python binding via pybind11

---

## License

GNU Lesser General Public License v3.0 — see [LICENSE](LICENSE).

Copyright (c) 2012-2024 Vissarion Fisikopoulos, Apostolos Chalkis, Elias Tsigaridas
and contributors.
