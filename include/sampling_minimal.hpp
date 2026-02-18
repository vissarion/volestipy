// Minimal sampling header for volestipy.
// Includes only uniform_sampling, gaussian_sampling, and
// exponential_sampling, avoiding the ifopt / ode_solvers /
// nlp_hpolyoracles dependency that the full sampling/sampling.hpp pulls in.

#pragma once

#include "volume/sampling_policies.hpp"
#include "sampling/random_point_generators.hpp"
#include "random_walks/exponential_hamiltonian_monte_carlo_exact_walk.hpp"

// ---------------------------------------------------------------------------
// uniform_sampling
// ---------------------------------------------------------------------------
template <typename WalkTypePolicy,
          typename PointList,
          typename Polytope,
          typename RandomNumberGenerator,
          typename Point>
void uniform_sampling(PointList &randPoints,
                      Polytope &P,
                      RandomNumberGenerator &rng,
                      const unsigned int &walk_len,
                      const unsigned int &rnum,
                      const Point &starting_point,
                      unsigned int const &nburns)
{
    typedef typename WalkTypePolicy::template Walk<Polytope, RandomNumberGenerator> walk;
    PushBackWalkPolicy push_back_policy;
    Point p = starting_point;
    typedef ::RandomPointGenerator<walk> RPG;
    if (nburns > 0) {
        RPG::apply(P, p, nburns, walk_len, randPoints, push_back_policy, rng);
        randPoints.clear();
    }
    RPG::apply(P, p, rnum, walk_len, randPoints, push_back_policy, rng);
}

// ---------------------------------------------------------------------------
// gaussian_sampling
// ---------------------------------------------------------------------------
template <typename WalkTypePolicy,
          typename PointList,
          typename Polytope,
          typename RandomNumberGenerator,
          typename NT,
          typename Point>
void gaussian_sampling(PointList &randPoints,
                       Polytope &P,
                       RandomNumberGenerator &rng,
                       const unsigned int &walk_len,
                       const unsigned int &rnum,
                       const NT &a,
                       const Point &starting_point,
                       unsigned int const &nburns)
{
    typedef typename WalkTypePolicy::template Walk<Polytope, RandomNumberGenerator> walk;
    PushBackWalkPolicy push_back_policy;
    Point p = starting_point;
    typedef GaussianRandomPointGenerator<walk> RPG;
    if (nburns > 0) {
        RPG::apply(P, p, a, nburns, walk_len, randPoints, push_back_policy, rng);
        randPoints.clear();
    }
    RPG::apply(P, p, a, rnum, walk_len, randPoints, push_back_policy, rng);
}

// ---------------------------------------------------------------------------
// exponential_sampling (via ExponentialRandomPointGenerator)
// Signature matches volesti sampling/sampling.hpp:
//   (randPoints, P, rng, walk_len, rnum, c, a, starting_point, nburns)
// ---------------------------------------------------------------------------
template <typename WalkTypePolicy,
          typename PointList,
          typename Polytope,
          typename RandomNumberGenerator,
          typename NT,
          typename Point>
void exponential_sampling(PointList &randPoints,
                          Polytope &P,
                          RandomNumberGenerator &rng,
                          const unsigned int &walk_len,
                          const unsigned int &rnum,
                          const Point &c,
                          const NT &a,
                          const Point &starting_point,
                          unsigned int const &nburns)
{
    typedef typename WalkTypePolicy::template Walk<Polytope, RandomNumberGenerator> walk;
    PushBackWalkPolicy push_back_policy;
    Point p = starting_point;
    typedef ExponentialRandomPointGenerator<walk> RPG;
    if (nburns > 0) {
        RPG::apply(P, p, c, a, nburns, walk_len, randPoints, push_back_policy, rng);
        randPoints.clear();
    }
    RPG::apply(P, p, c, a, rnum, walk_len, randPoints, push_back_policy, rng);
}
