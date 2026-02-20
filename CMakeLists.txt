# volestipy : a python library for sampling and volume computation
# volestipy is part of GeomScale project

# Licensed under GNU LGPL 2.1, see LICENCE file

cmake_minimum_required(VERSION 3.15)
project(volestipy VERSION 0.1.0 LANGUAGES C CXX)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build type ──────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Options ─────────────────────────────────────────────────────────────────
option(DISABLE_LPSOLVE "Disable the LP-solve dependency" OFF)
option(VOLESTIPY_USE_MKL "Use Intel MKL for Eigen" OFF)

# ─── Find pybind11 ───────────────────────────────────────────────────────────
# First try the Python-installed pybind11, then CMake config
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    find_package(pybind11 REQUIRED HINTS "${pybind11_DIR}")
endif()

# ─── Find Eigen3 ─────────────────────────────────────────────────────────────
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    # Fallback: look for Eigen in common locations
    find_path(EIGEN3_INCLUDE_DIR NAMES signature_of_eigen3_matrix_library
        PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen
        REQUIRED
    )
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# ─── Find Boost ──────────────────────────────────────────────────────────────
# volesti only needs Boost headers (random, math) - no compiled libraries required
find_package(Boost 1.56 REQUIRED)
if(NOT Boost_FOUND)
    # Fallback: locate headers manually
    find_path(Boost_INCLUDE_DIRS NAMES boost/random.hpp
        PATHS /usr/include /usr/local/include
        REQUIRED)
    message(STATUS "Boost headers found at ${Boost_INCLUDE_DIRS}")
endif()

# ─── volesti include path ─────────────────────────────────────────────────────
# You can override this with -DVOLESTI_INCLUDE_DIR=/path/to/volesti/include
set(VOLESTI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/volesti/include"
    CACHE PATH "Path to the volesti include directory")

if(NOT EXISTS "${VOLESTI_INCLUDE_DIR}/cartesian_geom/cartesian_kernel.h")
    message(FATAL_ERROR
        "Could not find volesti headers at '${VOLESTI_INCLUDE_DIR}'.\n"
        "Please either:\n"
        "  1. Run:  git submodule update --init  (if volesti is a submodule)\n"
        "  2. Set:  -DVOLESTI_INCLUDE_DIR=/path/to/volesti/include\n")
endif()

# ─── LP-solve (bundled source, same approach as dingo) ───────────────────────
set(LPSOLVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lp_solve_5.5")

if(DISABLE_LPSOLVE)
    set(LPSOLVE_DEFINITIONS -DDISABLE_LPSOLVE)
    set(LPSOLVE_SOURCES "")
    set(LPSOLVE_INCLUDE_DIRS "")
    message(STATUS "LP-solve disabled.")
elseif(EXISTS "${LPSOLVE_DIR}/lp_lib.h")
    message(STATUS "LP-solve bundled sources found at ${LPSOLVE_DIR}")
    set(LPSOLVE_DEFINITIONS
        -DYY_NEVER_INTERACTIVE
        -DLoadInverseLib=0
        -DLoadLanguageLib=0
        -DRoleIsExternalInvEngine
        -DINVERSE_ACTIVE=3
        -DLoadableBlasLib=0
    )
    set(LPSOLVE_SOURCES
        ${LPSOLVE_DIR}/bfp/bfp_LUSOL/lp_LUSOL.c
        ${LPSOLVE_DIR}/bfp/bfp_LUSOL/LUSOL/lusol.c
        ${LPSOLVE_DIR}/colamd/colamd.c
        ${LPSOLVE_DIR}/ini.c
        ${LPSOLVE_DIR}/shared/commonlib.c
        ${LPSOLVE_DIR}/shared/mmio.c
        ${LPSOLVE_DIR}/shared/myblas.c
        ${LPSOLVE_DIR}/lp_crash.c
        ${LPSOLVE_DIR}/lp_Hash.c
        ${LPSOLVE_DIR}/lp_lib.c
        ${LPSOLVE_DIR}/lp_matrix.c
        ${LPSOLVE_DIR}/lp_MDO.c
        ${LPSOLVE_DIR}/lp_mipbb.c
        ${LPSOLVE_DIR}/lp_MPS.c
        ${LPSOLVE_DIR}/lp_params.c
        ${LPSOLVE_DIR}/lp_presolve.c
        ${LPSOLVE_DIR}/lp_price.c
        ${LPSOLVE_DIR}/lp_pricePSE.c
        ${LPSOLVE_DIR}/lp_report.c
        ${LPSOLVE_DIR}/lp_scale.c
        ${LPSOLVE_DIR}/lp_simplex.c
        ${LPSOLVE_DIR}/lp_SOS.c
        ${LPSOLVE_DIR}/lp_utils.c
        ${LPSOLVE_DIR}/lp_wlp.c
    )
    set(LPSOLVE_INCLUDE_DIRS
        ${LPSOLVE_DIR}
        ${LPSOLVE_DIR}/bfp
        ${LPSOLVE_DIR}/bfp/bfp_LUSOL
        ${LPSOLVE_DIR}/bfp/bfp_LUSOL/LUSOL
        ${LPSOLVE_DIR}/colamd
        ${LPSOLVE_DIR}/shared
    )
else()
    message(WARNING "LP-solve bundled sources not found at ${LPSOLVE_DIR}; disabling.")
    message(WARNING "Run: wget -P external/ https://sourceforge.net/projects/lpsolve/files/lpsolve/5.5.2.11/lp_solve_5.5.2.11_source.tar.gz && tar xzf external/lp_solve_5.5.2.11_source.tar.gz -C external/")
    set(LPSOLVE_DEFINITIONS -DDISABLE_LPSOLVE)
    set(LPSOLVE_SOURCES "")
    set(LPSOLVE_INCLUDE_DIRS "")
endif()

# ─── Compiler flags ──────────────────────────────────────────────────────────
set(VOLESTIPY_COMPILE_OPTIONS
    -O3
    -march=native
    -funroll-loops
    -ffast-math
)

if(VOLESTIPY_USE_MKL)
    find_package(MKL REQUIRED)
    list(APPEND VOLESTIPY_COMPILE_OPTIONS -DEIGEN_USE_MKL_ALL)
endif()

# ─── Build the extension ─────────────────────────────────────────────────────
pybind11_add_module(_volestipy MODULE
    src/bindings/volesti_bindings.cpp
    ${LPSOLVE_SOURCES}
)

target_include_directories(_volestipy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # project local headers (sampling_minimal.hpp)
    ${VOLESTI_INCLUDE_DIR}
    ${VOLESTI_INCLUDE_DIR}/../external   # Spectra, minimum_ellipsoid, etc.
    ${Boost_INCLUDE_DIRS}
    ${LPSOLVE_INCLUDE_DIRS}
)

target_link_libraries(_volestipy PRIVATE
    Eigen3::Eigen
)

target_compile_definitions(_volestipy PRIVATE
    ${LPSOLVE_DEFINITIONS}
)

target_compile_options(_volestipy PRIVATE
    ${VOLESTIPY_COMPILE_OPTIONS}
)

# ─── Installation ────────────────────────────────────────────────────────────
install(TARGETS _volestipy
    LIBRARY DESTINATION volestipy
    RUNTIME DESTINATION volestipy
)

# ─── Status messages ─────────────────────────────────────────────────────────
message(STATUS "=== volestipy configuration ===")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  volesti headers  : ${VOLESTI_INCLUDE_DIR}")
message(STATUS "  Boost            : ${Boost_VERSION} (headers-only)")
message(STATUS "  pybind11         : ${pybind11_VERSION}")
if(DISABLE_LPSOLVE OR NOT EXISTS "${LPSOLVE_DIR}/lp_lib.h")
    message(STATUS "  LP-solve         : OFF")
else()
    message(STATUS "  LP-solve         : bundled (${LPSOLVE_DIR})")
endif()
message(STATUS "  Native arch      : ${VOLESTIPY_NATIVE_ARCH}")
message(STATUS "================================")
