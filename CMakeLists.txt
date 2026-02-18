cmake_minimum_required(VERSION 3.15)
project(volestipy VERSION 0.1.0 LANGUAGES CXX)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build type ──────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Options ─────────────────────────────────────────────────────────────────
option(DISABLE_LPSOLVE "Disable the LP-solve dependency" ON)
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
find_package(Boost 1.56 REQUIRED COMPONENTS
    random
    system
    filesystem
    math_c99
)

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

# ─── LP-solve (optional) ─────────────────────────────────────────────────────
if(DISABLE_LPSOLVE)
    set(LPSOLVE_DEFINITIONS -DDISABLE_LPSOLVE)
    message(STATUS "LP-solve disabled.")
else()
    find_library(LPSOLVE_LIB NAMES lpsolve55 lp_solve55)
    find_path(LPSOLVE_INCLUDE_DIR NAMES lp_lib.h)
    if(LPSOLVE_LIB AND LPSOLVE_INCLUDE_DIR)
        message(STATUS "LP-solve found: ${LPSOLVE_LIB}")
        set(LPSOLVE_DEFINITIONS "")
        set(LPSOLVE_LIBRARIES ${LPSOLVE_LIB})
    else()
        message(WARNING "LP-solve not found; disabling.")
        set(LPSOLVE_DEFINITIONS -DDISABLE_LPSOLVE)
    endif()
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
)

target_include_directories(_volestipy PRIVATE
    ${VOLESTI_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(_volestipy PRIVATE
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    ${LPSOLVE_LIBRARIES}
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
message(STATUS "  Eigen3           : ${EIGEN3_VERSION}")
message(STATUS "  Boost            : ${Boost_VERSION}")
message(STATUS "  pybind11         : ${pybind11_VERSION}")
message(STATUS "  LP-solve         : ${DISABLE_LPSOLVE}")
message(STATUS "================================")
