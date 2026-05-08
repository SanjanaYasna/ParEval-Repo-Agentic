// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion task/future model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include <boost/math/special_functions/sign.hpp>

#include "shared_resize.hpp"

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y) {
    if (x == 0.0) return 0.0;
    using std::abs;
    using std::pow;
    return pow(abs(x), y);
}
}  // namespace checked_math

inline double signed_pow(double x, double k) {
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

inline void system_first_block(const dvec& q, const double q_r, dvec& dpdt) {
    const std::size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
}

inline void system_center_block(const dvec& q, const double q_l, const double q_r, dvec& dpdt) {
    const std::size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
}

inline void system_last_block(const dvec& q, const double q_l, dvec& dpdt) {
    const std::size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
    for (std::size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1.0);
}

inline void osc_chain(const state_type& q, state_type& dpdt) {
    const std::size_t blocks = q.size();
    if (dpdt.size() != blocks) dpdt.resize(blocks);
    if (blocks == 0) return;

    for (std::size_t i = 0; i < blocks; ++i) {
        assert(q[i] && "q block must be initialized.");
        if (!dpdt[i]) dpdt[i] = std::make_shared<dvec>(q[i]->size());
    }

    if (blocks == 1) {
        system_first_block(*q[0], 0.0, *dpdt[0]);
        return;
    }

    system_first_block(*q[0], q[1]->front(), *dpdt[0]);

    for (std::size_t i = 1; i + 1 < blocks; ++i) {
        system_center_block(*q[i], q[i - 1]->back(), q[i + 1]->front(), *dpdt[i]);
    }

    system_last_block(*q[blocks - 1], q[blocks - 2]->back(), *dpdt[blocks - 1]);
}

inline void osc_chain_gb(const state_type& q, state_type& dpdt) {
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p) {
    if (q.empty()) return 0.0;

    using checked_math::pow;
    using std::abs;

    const std::size_t N = q.size();
    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i + 1 < N; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

inline double energy(const state_type& q_blocks, const state_type& p_blocks) {
    assert(q_blocks.size() == p_blocks.size());

    std::size_t total = 0;
    for (std::size_t i = 0; i < q_blocks.size(); ++i) {
        if (q_blocks[i]) total += q_blocks[i]->size();
    }

    dvec q;
    dvec p;
    q.reserve(total);
    p.reserve(total);

    for (std::size_t i = 0; i < q_blocks.size(); ++i) {
        assert(q_blocks[i] && p_blocks[i] && "state blocks must be initialized.");
        q.insert(q.end(), q_blocks[i]->begin(), q_blocks[i]->end());
        p.insert(p.end(), p_blocks[i]->begin(), p_blocks[i]->end());
    }

    return energy(q, p);
}

#endif
