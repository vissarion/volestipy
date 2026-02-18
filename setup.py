"""
setup.py for volestipy – Python bindings for the volesti library.

Build:
    pip install . --no-build-isolation
or:
    python setup.py build_ext --inplace
"""
import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# ── CMake-based build extension ──────────────────────────────────────────────

class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = "."):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_full_path = Path(self.get_ext_fullpath(ext.name))
        build_tmp = Path(self.build_temp)

        ext_dir = ext_full_path.parent.absolute()
        ext_dir.mkdir(parents=True, exist_ok=True)
        build_tmp.mkdir(parents=True, exist_ok=True)

        # Detect volesti include directory
        volesti_include = os.environ.get(
            "VOLESTI_INCLUDE_DIR",
            str(Path(ext.source_dir) / "external" / "volesti" / "include"),
        )

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DVOLESTI_INCLUDE_DIR={volesti_include}",
            f"-DCMAKE_BUILD_TYPE=Release",
            "-DDISABLE_LPSOLVE=ON",
        ]

        build_args = ["--config", "Release", "--", "-j4"]

        # Windows uses a different generator
        if sys.platform == "win32":
            cmake_args += ["-A", "x64"]
            build_args = ["--config", "Release"]

        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args, cwd=str(build_tmp)
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=str(build_tmp)
        )


# ── Package metadata ─────────────────────────────────────────────────────────

this_dir = Path(__file__).parent

long_description = (this_dir / "README.md").read_text(encoding="utf-8") \
    if (this_dir / "README.md").exists() else ""

setup(
    name="volestipy",
    version="0.1.0",
    author="volestipy contributors",
    description="Python bindings for the volesti library (convex body sampling & volume)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeomScale/volesti",
    license="LGPL-3.0-or-later",
    packages=["volestipy"],
    package_dir={"volestipy": "volestipy"},
    ext_modules=[CMakeExtension("volestipy._volestipy")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "matplotlib",
            "scipy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
)
