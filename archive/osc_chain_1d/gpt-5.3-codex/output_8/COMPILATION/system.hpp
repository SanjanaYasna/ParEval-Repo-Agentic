// Copyright 2013 Mario Mulansky
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

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
    if (x == 0.0) return 0.0;
    return std::copysign(std::pow(std::abs(x), k), x);
}

inline void compute_block_force(const dvec& q, double q_l, double q_r, dvec& dpdt) {
    const std::size_t n = q.size();
    dpdt.resize(n);
    if (n == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[n - 1] = -signed_pow(q[n - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[n - 1] - q_r, LAMBDA - 1.0);
}

inline void osc_chain(const state_type& q, state_type& dpdt) {
    const std::size_t num_blocks = q.size();
    if (dpdt.size() != num_blocks) dpdt.resize(num_blocks);
    if (num_blocks == 0) return;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const dvec& qb = *q[b];
        const std::size_t n = qb.size();

        if (!dpdt[b]) dpdt[b] = std::make_shared<dvec>(n);
        dvec& out = *dpdt[b];

        const double q_l = (b == 0 || !q[b - 1] || q[b - 1]->empty())
            ? 0.0
            : q[b - 1]->back();

        const double q_r = (b + 1 >= num_blocks || !q[b + 1] || q[b + 1]->empty())
            ? 0.0
            : q[b + 1]->front();

        compute_block_force(qb, q_l, q_r, out);
    }
}

inline void osc_chain_gb(const state_type& q, state_type& dpdt) {
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p) {
    using checked_math::pow;
    using std::abs;

    const std::size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i + 1 < N; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

template <typename QState, typename PState>
inline double energy(const QState& q_blocks, const PState& p_blocks) {
    dvec q, p;
    for (std::size_t i = 0; i < q_blocks.size(); ++i) {
        const dvec& q_block = *q_blocks[i];
        const dvec& p_block = *p_blocks[i];
        q.insert(q.end(), q_block.begin(), q_block.end());
        p.insert(p.end(), p_block.begin(), p_block.end());
    }
    return energy(q, p);
}

#endif
