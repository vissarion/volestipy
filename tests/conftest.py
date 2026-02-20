# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

# conftest.py - shared pytest fixtures for volestipy tests
import pytest
import numpy as np


@pytest.fixture
def cube_3d():
    """Return a 3-D hypercube [-1,1]^3 as an HPolytope."""
    try:
        from volestipy import hypercube
        return hypercube(3)
    except ImportError:
        pytest.skip("C++ extension not available")


@pytest.fixture
def simplex_3d():
    """Return the 3-D standard simplex as an HPolytope."""
    try:
        from volestipy import hypersimplex
        return hypersimplex(3)
    except ImportError:
        pytest.skip("C++ extension not available")


@pytest.fixture
def triangle_vpoly():
    """Return the 2-D triangle with vertices (0,0),(1,0),(0,1) as a VPolytope."""
    try:
        from volestipy import VPolytope
        V = np.array([[0., 0.], [1., 0.], [0., 1.]])
        return VPolytope(V)
    except ImportError:
        pytest.skip("C++ extension not available")
