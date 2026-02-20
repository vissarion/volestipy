// volestipy : a python library for sampling and volume computation
// volestipy is part of GeomScale project

// Licensed under GNU LGPL 2.1, see LICENCE file

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Disable LP-solve by default (optional dependency)
// #define DISABLE_LPSOLVE

// Volesti core headers
#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/vpolytope.h"
#include "convex_bodies/ball.h"
#include "convex_bodies/ballintersectconvex.h"

// Random walks (include only what we use, avoiding ifopt/nlp dependency)
#include "random_walks/uniform_cdhr_walk.hpp"
#include "random_walks/uniform_rdhr_walk.hpp"
#include "random_walks/uniform_ball_walk.hpp"
#include "random_walks/uniform_billiard_walk.hpp"
#include "random_walks/uniform_accelerated_billiard_walk.hpp"
#include "random_walks/uniform_dikin_walk.hpp"
#include "random_walks/uniform_john_walk.hpp"
#include "random_walks/uniform_vaidya_walk.hpp"
#include "random_walks/gaussian_cdhr_walk.hpp"
#include "random_walks/gaussian_rdhr_walk.hpp"
#include "random_walks/gaussian_ball_walk.hpp"
#include "random_walks/gaussian_helpers.hpp"

// Volume computation
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"

// Sampling (minimal version that avoids ifopt/ode_solvers dependency)
#include "sampling_minimal.hpp"

// Generators
#include "generators/boost_random_number_generator.hpp"
#include "generators/known_polytope_generators.h"

// Preprocessing
#include "preprocess/min_sampling_covering_ellipsoid_rounding.hpp"
#include "preprocess/inscribed_ellipsoid_rounding.hpp"

#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// Type aliases
typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> HPolytopeType;
typedef VPolytope<Point> VPolytopeType;
typedef Ball<Point> BallType;
typedef BoostRandomNumberGenerator<boost::mt11213b, NT> RNGType;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VectorXd;


// ============================================================
// Sampling helper: convert list<Point> -> Eigen matrix (points as columns)
// ============================================================
MatrixXd points_to_matrix(const std::list<Point>& pts) {
    if (pts.empty()) return MatrixXd(0, 0);
    int d = pts.begin()->dimension();
    int n = pts.size();
    MatrixXd M(d, n);
    int col = 0;
    for (const auto& p : pts)
        M.col(col++) = p.getCoefficients();
    return M;
}

// ============================================================
// Uniform sampling dispatcher
// ============================================================
MatrixXd hpoly_uniform_sample(HPolytopeType& P,
                               int n_samples,
                               int walk_length,
                               int n_burns,
                               const std::string& walk_type,
                               unsigned int seed) {
    RNGType rng(P.dimension());
    rng.set_seed(seed);

    Point starting_point(P.dimension());
    auto inner = P.ComputeInnerBall();
    if (inner.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball for HPolytope.");
    starting_point = inner.first;

    std::list<Point> randPoints;

    if (walk_type == "cdhr" || walk_type == "CDHR") {
        uniform_sampling<CDHRWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        uniform_sampling<RDHRWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        uniform_sampling<BallWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "billiard" || walk_type == "AcceleratedBilliardWalk") {
        uniform_sampling<BilliardWalk>(randPoints, P, rng,
                                       walk_length, n_samples,
                                       starting_point, n_burns);
    } else if (walk_type == "dikin" || walk_type == "DikinWalk") {
        uniform_sampling<DikinWalk>(randPoints, P, rng,
                                    walk_length, n_samples,
                                    starting_point, n_burns);
    } else if (walk_type == "john" || walk_type == "JohnWalk") {
        uniform_sampling<JohnWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "vaidya" || walk_type == "VaidyaWalk") {
        uniform_sampling<VaidyaWalk>(randPoints, P, rng,
                                     walk_length, n_samples,
                                     starting_point, n_burns);
    } else {
        throw std::invalid_argument("Unknown walk type: '" + walk_type +
            "'. Choose from: cdhr, rdhr, ball_walk, billiard, "
            "dikin, john, vaidya.");
    }

    return points_to_matrix(randPoints);
}

MatrixXd vpoly_uniform_sample(VPolytopeType& P,
                               int n_samples,
                               int walk_length,
                               int n_burns,
                               const std::string& walk_type,
                               unsigned int seed) {
    RNGType rng(P.dimension());
    rng.set_seed(seed);

    Point starting_point(P.dimension());
    auto inner = P.ComputeInnerBall();
    if (inner.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball for VPolytope.");
    starting_point = inner.first;

    std::list<Point> randPoints;

    if (walk_type == "cdhr" || walk_type == "CDHR") {
        uniform_sampling<CDHRWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        uniform_sampling<RDHRWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        uniform_sampling<BallWalk>(randPoints, P, rng,
                                   walk_length, n_samples,
                                   starting_point, n_burns);
    } else if (walk_type == "billiard" || walk_type == "BilliardWalk") {
        uniform_sampling<BilliardWalk>(randPoints, P, rng,
                                       walk_length, n_samples,
                                       starting_point, n_burns);
    } else {
        throw std::invalid_argument("Unknown walk type for VPolytope: '" + walk_type +
            "'. Choose from: cdhr, rdhr, ball_walk, billiard.");
    }

    return points_to_matrix(randPoints);
}

// ============================================================
// Gaussian sampling for HPolytope
// ============================================================
MatrixXd hpoly_gaussian_sample(HPolytopeType& P,
                                int n_samples,
                                int walk_length,
                                int n_burns,
                                double a,
                                const std::string& walk_type,
                                unsigned int seed) {
    RNGType rng(P.dimension());
    rng.set_seed(seed);

    auto inner = P.ComputeInnerBall();
    if (inner.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball for HPolytope.");
    Point starting_point = inner.first;

    std::list<Point> randPoints;

    if (walk_type == "cdhr" || walk_type == "CDHR") {
        gaussian_sampling<GaussianCDHRWalk>(randPoints, P, rng,
                                            walk_length, n_samples,
                                            a, starting_point, n_burns);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        gaussian_sampling<GaussianRDHRWalk>(randPoints, P, rng,
                                            walk_length, n_samples,
                                            a, starting_point, n_burns);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        gaussian_sampling<GaussianBallWalk>(randPoints, P, rng,
                                            walk_length, n_samples,
                                            a, starting_point, n_burns);
    } else {
        throw std::invalid_argument("Unknown Gaussian walk type: '" + walk_type +
            "'. Choose from: cdhr, rdhr, ball_walk.");
    }

    return points_to_matrix(randPoints);
}

// ============================================================
// Exponential sampling for HPolytope
// ============================================================
MatrixXd hpoly_exponential_sample(HPolytopeType& P,
                                   int n_samples,
                                   int walk_length,
                                   int n_burns,
                                   const VectorXd& c,
                                   double a,
                                   const std::string& walk_type,
                                   unsigned int seed) {
    RNGType rng(P.dimension());
    rng.set_seed(seed);

    auto inner = P.ComputeInnerBall();
    if (inner.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball for HPolytope.");
    Point starting_point = inner.first;
    Point bias_point(c);

    std::list<Point> randPoints;

    if (walk_type == "exponential_hmc" || walk_type == "ExponentialHMC") {
        exponential_sampling<ExponentialHamiltonianMonteCarloExactWalk>(
            randPoints, P, rng, walk_length, n_samples,
            bias_point, a, starting_point, n_burns);
    } else {
        // Default
        exponential_sampling<ExponentialHamiltonianMonteCarloExactWalk>(
            randPoints, P, rng, walk_length, n_samples,
            bias_point, a, starting_point, n_burns);
    }

    return points_to_matrix(randPoints);
}

// ============================================================
// Volume computation
// ============================================================
double hpoly_volume_sequence_of_balls(HPolytopeType& P,
                                       double error,
                                       int walk_length,
                                       const std::string& walk_type) {
    RNGType rng(P.dimension());
    double vol;
    if (walk_type == "cdhr" || walk_type == "CDHR") {
        vol = volume_sequence_of_balls<CDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        vol = volume_sequence_of_balls<RDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        vol = volume_sequence_of_balls<BallWalk>(P, rng, error, walk_length);
    } else if (walk_type == "billiard" || walk_type == "BilliardWalk") {
        vol = volume_sequence_of_balls<BilliardWalk>(P, rng, error, walk_length);
    } else {
        throw std::invalid_argument("Unknown walk type: '" + walk_type + "'.");
    }
    return vol;
}

double hpoly_volume_cooling_balls(HPolytopeType& P,
                                   double error,
                                   int walk_length,
                                   const std::string& walk_type) {
    RNGType rng(P.dimension());
    std::pair<double, double> res;
    if (walk_type == "cdhr" || walk_type == "CDHR") {
        res = volume_cooling_balls<CDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        res = volume_cooling_balls<RDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        res = volume_cooling_balls<BallWalk>(P, rng, error, walk_length);
    } else if (walk_type == "billiard" || walk_type == "BilliardWalk") {
        res = volume_cooling_balls<BilliardWalk>(P, rng, error, walk_length);
    } else {
        throw std::invalid_argument("Unknown walk type: '" + walk_type + "'.");
    }
    return res.second;  // return actual volume (not log)
}

double hpoly_volume_cooling_gaussians(HPolytopeType& P,
                                       double error,
                                       int walk_length,
                                       const std::string& walk_type) {
    RNGType rng(P.dimension());
    double vol;
    if (walk_type == "cdhr" || walk_type == "CDHR") {
        vol = volume_cooling_gaussians<GaussianCDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        vol = volume_cooling_gaussians<GaussianRDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        vol = volume_cooling_gaussians<GaussianBallWalk>(P, rng, error, walk_length);
    } else {
        throw std::invalid_argument("Unknown Gaussian walk type: '" + walk_type + "'.");
    }
    return vol;
}

// VPolytope volume
double vpoly_volume_sequence_of_balls(VPolytopeType& P,
                                       double error,
                                       int walk_length,
                                       const std::string& walk_type) {
    RNGType rng(P.dimension());
    double vol;
    if (walk_type == "cdhr" || walk_type == "CDHR") {
        vol = volume_sequence_of_balls<CDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        vol = volume_sequence_of_balls<RDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        vol = volume_sequence_of_balls<BallWalk>(P, rng, error, walk_length);
    } else {
        throw std::invalid_argument("Unknown walk type: '" + walk_type + "'.");
    }
    return vol;
}

double vpoly_volume_cooling_balls(VPolytopeType& P,
                                   double error,
                                   int walk_length,
                                   const std::string& walk_type) {
    RNGType rng(P.dimension());
    std::pair<double, double> res;
    if (walk_type == "cdhr" || walk_type == "CDHR") {
        res = volume_cooling_balls<CDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "rdhr" || walk_type == "RDHR") {
        res = volume_cooling_balls<RDHRWalk>(P, rng, error, walk_length);
    } else if (walk_type == "ball_walk" || walk_type == "BallWalk") {
        res = volume_cooling_balls<BallWalk>(P, rng, error, walk_length);
    } else {
        throw std::invalid_argument("Unknown walk type: '" + walk_type + "'.");
    }
    return res.second;
}

// ============================================================
// Rounding helpers
// ============================================================
py::tuple hpoly_round_min_ellipsoid(HPolytopeType& P,
                                     int walk_length,
                                     unsigned int seed) {
    auto inner_ball = P.ComputeInnerBall();
    if (inner_ball.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball.");
    RNGType rng(P.dimension());
    rng.set_seed(seed);
    auto [T, shift, round_val] =
        min_sampling_covering_ellipsoid_rounding<CDHRWalk, MatrixXd, VectorXd>(
            P, inner_ball, walk_length, rng);
    return py::make_tuple(T, shift, round_val);
}

py::tuple hpoly_round_max_inscribed_ellipsoid(HPolytopeType& P) {
    auto inner_ball = P.ComputeInnerBall();
    if (inner_ball.second < 0.0)
        throw std::runtime_error("Failed to compute inner ball.");
    auto [T, shift, round_val] =
        inscribed_ellipsoid_rounding<MatrixXd, VectorXd, NT>(P, inner_ball.first);
    return py::make_tuple(T, shift, round_val);
}


// ============================================================
// Module definition
// ============================================================
PYBIND11_MODULE(_volestipy, m) {
    m.doc() = R"pbdoc(
        volestipy - Python bindings for the volesti library.

        Provides:
          - HPolytope: polytope in H-representation (Ax <= b)
          - VPolytope: polytope in V-representation (convex hull of vertices)
          - Uniform, Gaussian, and Exponential sampling with multiple MCMC walks
          - Volume computation via sequence-of-balls, cooling-balls, and cooling-gaussians
          - Rounding utilities
    )pbdoc";

    // ----------------------------------------------------------
    // HPolytope
    // ----------------------------------------------------------
    py::class_<HPolytopeType>(m, "HPolytope",
        R"pbdoc(
        H-polytope: a convex polytope defined as { x : A x <= b }.

        Parameters
        ----------
        A : array-like of shape (m, d)
            Constraint matrix.
        b : array-like of shape (m,)
            Right-hand-side vector.
        )pbdoc")
        .def(py::init<>(), "Construct an empty H-polytope.")
        .def(py::init([](const MatrixXd& A, const VectorXd& b) {
            unsigned int d = A.cols();
            if (A.rows() != b.size())
                throw std::invalid_argument("A.rows() must equal b.size().");
            return new HPolytopeType(d, A, b);
        }), py::arg("A"), py::arg("b"),
        "Construct H-polytope from constraint matrix A and vector b (Ax <= b).")

        .def("dimension", &HPolytopeType::dimension,
             "Return the dimension of the polytope.")
        .def("num_of_hyperplanes", &HPolytopeType::num_of_hyperplanes,
             "Return the number of hyperplane constraints.")
        .def("get_mat", &HPolytopeType::get_mat,
             "Return the constraint matrix A.")
        .def("get_vec", &HPolytopeType::get_vec,
             "Return the right-hand-side vector b.")
        .def("set_mat", &HPolytopeType::set_mat, py::arg("A"),
             "Set the constraint matrix A.")
        .def("set_vec", &HPolytopeType::set_vec, py::arg("b"),
             "Set the right-hand-side vector b.")
        .def("is_in", [](const HPolytopeType& P, const VectorXd& p) {
            Point pt(p);
            return P.is_in(pt);
        }, py::arg("p"),
        "Return -1 if point p is in the polytope, 0 otherwise.")
        .def("compute_inner_ball", [](HPolytopeType& P) {
            auto res = P.ComputeInnerBall();
            return py::make_tuple(res.first.getCoefficients(), res.second);
        }, "Compute the largest inscribed ball. Returns (center, radius).")
        .def("normalize", &HPolytopeType::normalize,
             "Normalize the rows of A so each has unit norm.")
        .def("shift", [](HPolytopeType& P, const VectorXd& c) {
            P.shift(c);
        }, py::arg("c"), "Shift the polytope by vector c.")
        .def("print", &HPolytopeType::print, "Print the polytope.")

        // Sampling
        .def("uniform_sample", [](HPolytopeType& P,
                                   int n_samples,
                                   int walk_length,
                                   int n_burns,
                                   const std::string& walk_type,
                                   unsigned int seed) {
            return hpoly_uniform_sample(P, n_samples, walk_length,
                                        n_burns, walk_type, seed);
        }, py::arg("n_samples") = 1000,
           py::arg("walk_length") = 1,
           py::arg("n_burns") = 0,
           py::arg("walk_type") = "cdhr",
           py::arg("seed") = 0,
        R"pbdoc(
        Draw uniform samples from the polytope.

        Parameters
        ----------
        n_samples : int
            Number of sample points.
        walk_length : int
            Number of steps per sample (thinning).
        n_burns : int
            Number of burn-in steps.
        walk_type : str
            Walk algorithm. One of: 'cdhr', 'rdhr', 'ball_walk',
            'billiard', 'dikin', 'john', 'vaidya'.
        seed : int
            Random seed.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
            Sample points as columns.
        )pbdoc")

        .def("gaussian_sample", [](HPolytopeType& P,
                                    int n_samples,
                                    int walk_length,
                                    int n_burns,
                                    double a,
                                    const std::string& walk_type,
                                    unsigned int seed) {
            return hpoly_gaussian_sample(P, n_samples, walk_length,
                                         n_burns, a, walk_type, seed);
        }, py::arg("n_samples") = 1000,
           py::arg("walk_length") = 1,
           py::arg("n_burns") = 0,
           py::arg("a") = 1.0,
           py::arg("walk_type") = "cdhr",
           py::arg("seed") = 0,
        R"pbdoc(
        Draw samples from the Gaussian distribution exp(-a ||x||^2) restricted
        to the polytope.

        Parameters
        ----------
        n_samples : int
            Number of sample points.
        walk_length : int
            Thinning parameter.
        n_burns : int
            Burn-in steps.
        a : float
            Variance parameter (default 1.0).
        walk_type : str
            'cdhr', 'rdhr', or 'ball_walk'.
        seed : int
            Random seed.

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        )pbdoc")

        .def("exponential_sample", [](HPolytopeType& P,
                                       int n_samples,
                                       int walk_length,
                                       int n_burns,
                                       const VectorXd& c,
                                       double a,
                                       const std::string& walk_type,
                                       unsigned int seed) {
            return hpoly_exponential_sample(P, n_samples, walk_length,
                                            n_burns, c, a, walk_type, seed);
        }, py::arg("n_samples") = 1000,
           py::arg("walk_length") = 1,
           py::arg("n_burns") = 0,
           py::arg("c"),
           py::arg("a") = 1.0,
           py::arg("walk_type") = "exponential_hmc",
           py::arg("seed") = 0,
        R"pbdoc(
        Draw samples from the exponential distribution exp(a * c^T x)
        restricted to the polytope.

        Parameters
        ----------
        n_samples : int
        walk_length : int
        n_burns : int
        c : array-like of shape (d,)
            Bias direction vector.
        a : float
            Variance parameter.
        walk_type : str
            'exponential_hmc'.
        seed : int

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        )pbdoc")

        // Volume computation
        .def("volume", [](HPolytopeType& P,
                          double error,
                          int walk_length,
                          const std::string& algorithm,
                          const std::string& walk_type) {
            if (algorithm == "sequence_of_balls" || algorithm == "SOB") {
                return hpoly_volume_sequence_of_balls(P, error, walk_length, walk_type);
            } else if (algorithm == "cooling_balls" || algorithm == "CB") {
                return hpoly_volume_cooling_balls(P, error, walk_length, walk_type);
            } else if (algorithm == "cooling_gaussians" || algorithm == "CG") {
                return hpoly_volume_cooling_gaussians(P, error, walk_length, walk_type);
            } else {
                throw std::invalid_argument(
                    "Unknown algorithm: '" + algorithm + "'. "
                    "Choose from: 'sequence_of_balls', 'cooling_balls', 'cooling_gaussians'.");
            }
        }, py::arg("error") = 0.1,
           py::arg("walk_length") = 1,
           py::arg("algorithm") = "cooling_balls",
           py::arg("walk_type") = "cdhr",
        R"pbdoc(
        Estimate the volume of the polytope.

        Parameters
        ----------
        error : float
            Relative error bound (default 0.1 = 10%).
        walk_length : int
            Steps per sample.
        algorithm : str
            Volume algorithm: 'sequence_of_balls', 'cooling_balls',
            'cooling_gaussians'.
        walk_type : str
            MCMC walk: 'cdhr', 'rdhr', 'ball_walk', 'billiard'.

        Returns
        -------
        float
            Estimated volume.
        )pbdoc")

        // Rounding
        .def("round_min_ellipsoid", [](HPolytopeType& P,
                                       int walk_length,
                                       unsigned int seed) {
            return hpoly_round_min_ellipsoid(P, walk_length, seed);
        }, py::arg("walk_length") = 1, py::arg("seed") = 0,
        R"pbdoc(
        Round the polytope via the minimum enclosing ellipsoid of random samples.
        Returns (T, T_shift, round_value).
        )pbdoc")
        .def("round_max_ellipsoid", [](HPolytopeType& P) {
            return hpoly_round_max_inscribed_ellipsoid(P);
        }, R"pbdoc(
        Round the polytope by mapping the maximum inscribed ellipsoid to the
        unit ball. Returns (T, T_shift, round_value).
        )pbdoc");

    // ----------------------------------------------------------
    // VPolytope
    // ----------------------------------------------------------
    py::class_<VPolytopeType>(m, "VPolytope",
        R"pbdoc(
        V-polytope: a convex polytope defined as the convex hull of vertices.

        Parameters
        ----------
        V : array-like of shape (n_vertices, d)
            Matrix whose rows are the vertices.
        )pbdoc")
        .def(py::init<>(), "Construct an empty V-polytope.")
        .def(py::init([](const MatrixXd& V) {
            unsigned int d = V.cols();
            VectorXd b = VectorXd::Ones(V.rows());  // placeholder
            return new VPolytopeType(d, V, b);
        }), py::arg("V"),
        "Construct V-polytope from vertex matrix V (rows are vertices).")

        .def("dimension", &VPolytopeType::dimension,
             "Return the dimension of the polytope.")
        .def("num_of_vertices", &VPolytopeType::num_of_vertices,
             "Return the number of vertices.")
        .def("get_mat", &VPolytopeType::get_mat,
             "Return the vertex matrix V.")
        .def("set_mat", &VPolytopeType::set_mat, py::arg("V"),
             "Set the vertex matrix V.")
        .def("is_in", [](const VPolytopeType& P, const VectorXd& p) {
            Point pt(p);
            return P.is_in(pt);
        }, py::arg("p"),
        "Return -1 if point p is in the polytope, 0 otherwise.")
        .def("compute_inner_ball", [](VPolytopeType& P) {
            auto res = P.ComputeInnerBall();
            return py::make_tuple(res.first.getCoefficients(), res.second);
        }, "Compute an inscribed ball. Returns (center, radius).")
        .def("print", &VPolytopeType::print, "Print the polytope.")
        .def("shift", [](VPolytopeType& P, const VectorXd& c) {
            P.shift(c);
        }, py::arg("c"), "Shift the polytope by vector c.")

        // Sampling
        .def("uniform_sample", [](VPolytopeType& P,
                                   int n_samples,
                                   int walk_length,
                                   int n_burns,
                                   const std::string& walk_type,
                                   unsigned int seed) {
            return vpoly_uniform_sample(P, n_samples, walk_length,
                                        n_burns, walk_type, seed);
        }, py::arg("n_samples") = 1000,
           py::arg("walk_length") = 1,
           py::arg("n_burns") = 0,
           py::arg("walk_type") = "cdhr",
           py::arg("seed") = 0,
        R"pbdoc(
        Draw uniform samples from the V-polytope.

        Parameters
        ----------
        n_samples : int
        walk_length : int
        n_burns : int
        walk_type : str
            'cdhr', 'rdhr', 'ball_walk', 'billiard'.
        seed : int

        Returns
        -------
        numpy.ndarray of shape (d, n_samples)
        )pbdoc")

        // Volume
        .def("volume", [](VPolytopeType& P,
                          double error,
                          int walk_length,
                          const std::string& algorithm,
                          const std::string& walk_type) {
            if (algorithm == "sequence_of_balls" || algorithm == "SOB") {
                return vpoly_volume_sequence_of_balls(P, error, walk_length, walk_type);
            } else if (algorithm == "cooling_balls" || algorithm == "CB") {
                return vpoly_volume_cooling_balls(P, error, walk_length, walk_type);
            } else {
                throw std::invalid_argument(
                    "Unknown algorithm: '" + algorithm + "'.");
            }
        }, py::arg("error") = 0.1,
           py::arg("walk_length") = 1,
           py::arg("algorithm") = "cooling_balls",
           py::arg("walk_type") = "cdhr",
        R"pbdoc(
        Estimate the volume of the V-polytope.

        Parameters
        ----------
        error : float
        walk_length : int
        algorithm : str
            'sequence_of_balls' or 'cooling_balls'.
        walk_type : str

        Returns
        -------
        float
        )pbdoc");

    // ----------------------------------------------------------
    // Module-level free functions
    // ----------------------------------------------------------
    m.def("hpoly_volume", [](const MatrixXd& A, const VectorXd& b,
                              double error, int walk_length,
                              const std::string& algorithm,
                              const std::string& walk_type) {
        unsigned int d = A.cols();
        HPolytopeType P(d, A, b);
        if (algorithm == "sequence_of_balls" || algorithm == "SOB") {
            return hpoly_volume_sequence_of_balls(P, error, walk_length, walk_type);
        } else if (algorithm == "cooling_balls" || algorithm == "CB") {
            return hpoly_volume_cooling_balls(P, error, walk_length, walk_type);
        } else if (algorithm == "cooling_gaussians" || algorithm == "CG") {
            return hpoly_volume_cooling_gaussians(P, error, walk_length, walk_type);
        }
        throw std::invalid_argument("Unknown algorithm: " + algorithm);
    }, py::arg("A"), py::arg("b"),
       py::arg("error") = 0.1,
       py::arg("walk_length") = 1,
       py::arg("algorithm") = "cooling_balls",
       py::arg("walk_type") = "cdhr",
    "Compute the volume of the H-polytope {x: Ax <= b}.");

    m.def("hpoly_sample", [](const MatrixXd& A, const VectorXd& b,
                              int n_samples, int walk_length, int n_burns,
                              const std::string& walk_type, unsigned int seed) {
        unsigned int d = A.cols();
        HPolytopeType P(d, A, b);
        return hpoly_uniform_sample(P, n_samples, walk_length, n_burns, walk_type, seed);
    }, py::arg("A"), py::arg("b"),
       py::arg("n_samples") = 1000,
       py::arg("walk_length") = 1,
       py::arg("n_burns") = 0,
       py::arg("walk_type") = "cdhr",
       py::arg("seed") = 0,
    "Uniformly sample from the H-polytope {x: Ax <= b}.");

    m.def("vpoly_sample", [](const MatrixXd& V,
                              int n_samples, int walk_length, int n_burns,
                              const std::string& walk_type, unsigned int seed) {
        unsigned int d = V.cols();
        VectorXd b = VectorXd::Ones(V.rows());
        VPolytopeType P(d, V, b);
        return vpoly_uniform_sample(P, n_samples, walk_length, n_burns, walk_type, seed);
    }, py::arg("V"),
       py::arg("n_samples") = 1000,
       py::arg("walk_length") = 1,
       py::arg("n_burns") = 0,
       py::arg("walk_type") = "cdhr",
       py::arg("seed") = 0,
    "Uniformly sample from the V-polytope (convex hull of rows of V).");

    m.def("vpoly_volume", [](const MatrixXd& V,
                              double error, int walk_length,
                              const std::string& algorithm,
                              const std::string& walk_type) {
        unsigned int d = V.cols();
        VectorXd b = VectorXd::Ones(V.rows());
        VPolytopeType P(d, V, b);
        if (algorithm == "sequence_of_balls" || algorithm == "SOB") {
            return vpoly_volume_sequence_of_balls(P, error, walk_length, walk_type);
        } else if (algorithm == "cooling_balls" || algorithm == "CB") {
            return vpoly_volume_cooling_balls(P, error, walk_length, walk_type);
        }
        throw std::invalid_argument("Unknown algorithm: " + algorithm);
    }, py::arg("V"),
       py::arg("error") = 0.1,
       py::arg("walk_length") = 1,
       py::arg("algorithm") = "cooling_balls",
       py::arg("walk_type") = "cdhr",
    "Compute the volume of the V-polytope (convex hull of rows of V).");

    // ── Birkhoff polytope generator ──────────────────────────────────────────
    m.def("hpoly_birkhoff", [](unsigned int n) {
        if (n < 2)
            throw std::invalid_argument(
                "Birkhoff polytope requires n >= 2 (n=1 is a single point).");
        HPolytopeType P = generate_birkhoff<HPolytopeType>(n);
        MatrixXd A = P.get_mat();
        VectorXd b = P.get_vec();
        return py::make_tuple(A, b);
    }, py::arg("n"),
    R"(Generate the Birkhoff polytope B(n) in H-representation.

The Birkhoff polytope B(n) is the convex polytope of n×n doubly
stochastic matrices (non-negative real entries whose rows and columns
each sum to 1).  It lives in dimension d = (n-1)^2.

Parameters
----------
n : int
    Matrix size (n >= 2).  B(2) is a line segment, B(3) has
    dimension 4, B(4) has dimension 9, etc.

Returns
-------
A : ndarray, shape (m, d)
    Constraint matrix.
b : ndarray, shape (m,)
    Right-hand side vector.  The polytope is { x : A x <= b }.
)");

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__volesti_version__") = "1.1.2";
}
