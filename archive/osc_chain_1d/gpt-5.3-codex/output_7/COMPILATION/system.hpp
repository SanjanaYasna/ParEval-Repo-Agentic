// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion task-launch model.

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstddef>

#include <boost/math/special_functions/sign.hpp>
#include <legion.h>

inline constexpr double KAPPA = 3.5;
inline constexpr double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y)
{
    if (x == 0.0) return 0.0;
    using std::abs;
    using std::pow;
    return pow(abs(x), y);
}
} // namespace checked_math

inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

using dvec = std::vector<double>;
using shared_vec = std::shared_ptr<dvec>;

// Legion translation uses concrete block storage (no HPX shared_future wrapper).
using state_type = std::vector<shared_vec>;

inline void compute_block_rhs(const dvec& q,
                              double q_l, double q_r,
                              bool has_left, bool has_right,
                              dvec& dpdt)
{
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    const double left_boundary  = has_left  ? q_l : 0.0;
    const double right_boundary = has_right ? q_r : 0.0;

    double coupling_lr = -signed_pow(q[0] - left_boundary, LAMBDA - 1);
    for (size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - right_boundary, LAMBDA - 1);
}

// Kept for compatibility with old call sites/structure.
struct system_first_block {
    shared_vec operator()(shared_vec q, const double q_r, shared_vec dpdt) const {
        if (!dpdt) dpdt = std::make_shared<dvec>();
        compute_block_rhs(*q, 0.0, q_r, false, true, *dpdt);
        return dpdt;
    }
};

struct system_center_block {
    shared_vec operator()(shared_vec q, const double q_l, const double q_r, shared_vec dpdt) const {
        if (!dpdt) dpdt = std::make_shared<dvec>();
        compute_block_rhs(*q, q_l, q_r, true, true, *dpdt);
        return dpdt;
    }
};

struct system_last_block {
    shared_vec operator()(shared_vec q, const double q_l, shared_vec dpdt) const {
        if (!dpdt) dpdt = std::make_shared<dvec>();
        compute_block_rhs(*q, q_l, 0.0, true, false, *dpdt);
        return dpdt;
    }
};

namespace legion_system_detail {

inline Legion::Runtime* g_runtime = nullptr;
inline Legion::Context  g_context = nullptr;

inline constexpr Legion::TaskID SYSTEM_BLOCK_TASK_ID = 10001;

struct BlockTaskArgs {
    const dvec* q_ptr;
    dvec*       dpdt_ptr;
    double      q_l;
    double      q_r;
    bool        has_left;
    bool        has_right;
};

inline void system_block_task(const Legion::Task* task,
                              const std::vector<Legion::PhysicalRegion>&,
                              Legion::Context,
                              Legion::Runtime*)
{
    const auto* args = static_cast<const BlockTaskArgs*>(task->args);
    if (!args || !args->q_ptr || !args->dpdt_ptr) return;

    compute_block_rhs(*args->q_ptr,
                      args->q_l, args->q_r,
                      args->has_left, args->has_right,
                      *args->dpdt_ptr);
}

inline void bind_runtime_context(Legion::Runtime* runtime, Legion::Context context)
{
    g_runtime = runtime;
    g_context = context;
}

// Call once before Runtime::start() in your Legion main/bootstrap code.
inline void preregister_system_tasks()
{
    static bool registered = false;
    if (registered) return;

    Legion::TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<system_block_task>(registrar, "system_block_task");

    registered = true;
}

} // namespace legion_system_detail

// Public helpers for the rest of the translated codebase.
inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context context)
{
    legion_system_detail::bind_runtime_context(runtime, context);
}

inline void preregister_system_tasks()
{
    legion_system_detail::preregister_system_tasks();
}

inline void osc_chain(state_type& q, state_type& dpdt)
{
    const size_t N = q.size();
    if (dpdt.size() != N) dpdt.resize(N);
    if (N == 0) return;

    // Fallback if context not bound yet: execute sequentially.
    if (legion_system_detail::g_runtime == nullptr || legion_system_detail::g_context == nullptr) {
        for (size_t i = 0; i < N; ++i) {
            if (!q[i]) q[i] = std::make_shared<dvec>();
            if (!dpdt[i]) dpdt[i] = std::make_shared<dvec>();

            const double q_l = (i > 0 && q[i - 1] && !q[i - 1]->empty()) ? q[i - 1]->back() : 0.0;
            const double q_r = (i + 1 < N && q[i + 1] && !q[i + 1]->empty()) ? q[i + 1]->front() : 0.0;

            compute_block_rhs(*q[i], q_l, q_r, i > 0, i + 1 < N, *dpdt[i]);
        }
        return;
    }

    std::vector<Legion::Future> done;
    done.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        if (!q[i]) q[i] = std::make_shared<dvec>();
        if (!dpdt[i]) dpdt[i] = std::make_shared<dvec>();

        const double q_l = (i > 0 && q[i - 1] && !q[i - 1]->empty()) ? q[i - 1]->back() : 0.0;
        const double q_r = (i + 1 < N && q[i + 1] && !q[i + 1]->empty()) ? q[i + 1]->front() : 0.0;

        legion_system_detail::BlockTaskArgs args{
            q[i].get(), dpdt[i].get(), q_l, q_r, (i > 0), (i + 1 < N)
        };

        Legion::TaskLauncher launcher(
            legion_system_detail::SYSTEM_BLOCK_TASK_ID,
            Legion::TaskArgument(&args, sizeof(args))
        );
        done.emplace_back(
            legion_system_detail::g_runtime->execute_task(legion_system_detail::g_context, launcher)
        );
    }

    // Ensure completion before returning (barrier semantics).
    for (auto& f : done) f.get_void_result();
}

inline void osc_chain_gb(state_type& q, state_type& dpdt)
{
    // osc_chain already enforces completion of all launched block tasks.
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p)
{
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
inline double energy(const S& q_blocks, const S& p_blocks)
{
    dvec q, p;
    for (size_t i = 0; i < q_blocks.size(); ++i) {
        q.insert(q.end(), q_blocks[i]->begin(), q_blocks[i]->end());
        p.insert(p.end(), p_blocks[i]->begin(), p_blocks[i]->end());
    }
    return energy(q, p);
}

#endif
