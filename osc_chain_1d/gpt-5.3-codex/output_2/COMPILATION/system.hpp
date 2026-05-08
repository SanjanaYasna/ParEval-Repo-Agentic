// Copyright 2013 Mario Mulansky
// Translated to Legion-oriented execution (default mapper, no custom mapping)

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <cstddef>
#include <algorithm>

#include <boost/math/special_functions/sign.hpp>
#include <legion.h>

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

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// Legion translation: explicit block storage (no HPX futures/dataflow)
typedef std::vector<shared_vec> state_type;

namespace legion_odeint_detail {

enum : Legion::TaskID {
    OSC_CHAIN_BLOCK_TASK_ID = 10001
};

enum BlockKind : int {
    FIRST_BLOCK = 0,
    CENTER_BLOCK = 1,
    LAST_BLOCK = 2,
    SINGLE_BLOCK = 3
};

struct BlockTaskArgs {
    const dvec* q;
    dvec* dpdt;
    double q_l;
    double q_r;
    int kind;
};

inline void ensure_block(const shared_vec& src, shared_vec& dst) {
    if (!dst) dst = std::make_shared<dvec>();
    if (!src) {
        dst->clear();
        return;
    }
    if (dst->size() != src->size()) dst->assign(src->size(), 0.0);
}

inline void compute_first_block(const dvec& q, double q_r, dvec& dpdt) {
    const size_t N = q.size();
    if (N == 0) return;
    if (dpdt.size() != N) dpdt.resize(N);

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_center_block(const dvec& q, double q_l, double q_r, dvec& dpdt) {
    const size_t N = q.size();
    if (N == 0) return;
    if (dpdt.size() != N) dpdt.resize(N);

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_last_block(const dvec& q, double q_l, dvec& dpdt) {
    const size_t N = q.size();
    if (N == 0) return;
    if (dpdt.size() != N) dpdt.resize(N);

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1);
}

inline void osc_chain_block_task(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>&,
                                 Legion::Context,
                                 Legion::Runtime*) {
    const auto* args = static_cast<const BlockTaskArgs*>(task->args);
    const dvec& q = *(args->q);
    dvec& dpdt = *(args->dpdt);

    switch (args->kind) {
        case FIRST_BLOCK:  compute_first_block(q, args->q_r, dpdt); break;
        case CENTER_BLOCK: compute_center_block(q, args->q_l, args->q_r, dpdt); break;
        case LAST_BLOCK:   compute_last_block(q, args->q_l, dpdt); break;
        case SINGLE_BLOCK: compute_center_block(q, 0.0, 0.0, dpdt); break;
        default:           compute_center_block(q, args->q_l, args->q_r, dpdt); break;
    }
}

inline bool preregister_tasks() {
    Legion::TaskVariantRegistrar registrar(OSC_CHAIN_BLOCK_TASK_ID, "osc_chain_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf(true);
    Legion::Runtime::preregister_task_variant<osc_chain_block_task>(registrar, "osc_chain_block_task");
    return true;
}

inline bool g_registered = preregister_tasks();
inline Legion::Runtime* g_runtime = nullptr;
inline Legion::Context g_context = nullptr;

}  // namespace legion_odeint_detail

// Call once from a Legion task context (e.g., top-level task) before integration
inline void set_legion_system_context(Legion::Runtime* runtime, Legion::Context context) {
    (void)legion_odeint_detail::g_registered;
    legion_odeint_detail::g_runtime = runtime;
    legion_odeint_detail::g_context = context;
}

inline void osc_chain(state_type& q, state_type& dpdt) {
    using namespace legion_odeint_detail;

    const size_t N = q.size();
    if (dpdt.size() != N) dpdt.resize(N);
    if (N == 0) return;

    for (size_t i = 0; i < N; ++i) ensure_block(q[i], dpdt[i]);

    // If Legion context is not set, run in-place serially (safe fallback)
    if (g_runtime == nullptr || g_context == nullptr) {
        if (N == 1) {
            compute_center_block(*q[0], 0.0, 0.0, *dpdt[0]);
            return;
        }

        compute_first_block(*q[0], (*q[1])[0], *dpdt[0]);
        for (size_t i = 1; i + 1 < N; ++i) {
            compute_center_block(*q[i], q[i - 1]->back(), (*q[i + 1])[0], *dpdt[i]);
        }
        compute_last_block(*q[N - 1], q[N - 2]->back(), *dpdt[N - 1]);
        return;
    }

    std::vector<Legion::Future> futures;
    futures.reserve(N);

    if (N == 1) {
        BlockTaskArgs args{q[0].get(), dpdt[0].get(), 0.0, 0.0, SINGLE_BLOCK};
        Legion::TaskLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
        futures.push_back(g_runtime->execute_task(g_context, launcher));
    } else {
        {
            BlockTaskArgs args{q[0].get(), dpdt[0].get(), 0.0, (*q[1])[0], FIRST_BLOCK};
            Legion::TaskLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
            futures.push_back(g_runtime->execute_task(g_context, launcher));
        }

        for (size_t i = 1; i + 1 < N; ++i) {
            BlockTaskArgs args{
                q[i].get(), dpdt[i].get(),
                q[i - 1]->back(), (*q[i + 1])[0],
                CENTER_BLOCK
            };
            Legion::TaskLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
            futures.push_back(g_runtime->execute_task(g_context, launcher));
        }

        {
            BlockTaskArgs args{q[N - 1].get(), dpdt[N - 1].get(), q[N - 2]->back(), 0.0, LAST_BLOCK};
            Legion::TaskLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
            futures.push_back(g_runtime->execute_task(g_context, launcher));
        }
    }

    for (auto& f : futures) f.get_void_result();
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
    // In Legion version, osc_chain already synchronizes task completion.
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p) {
    using checked_math::pow;
    using std::abs;

    const size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i + 1 < N; ++i) {
        e += 0.5 * p[i] * p[i]
           + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

template <typename S>
inline double energy(const S& q_blocks, const S& p_blocks) {
    dvec q;
    dvec p;

    for (size_t i = 0; i < q_blocks.size(); ++i) {
        if (q_blocks[i]) q.insert(q.end(), q_blocks[i]->begin(), q_blocks[i]->end());
        if (p_blocks[i]) p.insert(p.end(), p_blocks[i]->begin(), p_blocks[i]->end());
    }

    return energy(q, p);
}

#endif
