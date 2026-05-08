// Copyright 2013 Mario Mulansky
// Legion translation

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <boost/math/special_functions/sign.hpp>

#include <legion.h>
#if __has_include(<legion/legion_stl.h>)
#include <legion/legion_stl.h>
#elif __has_include(<legion_stl.h>)
#include <legion_stl.h>
#endif

const double KAPPA  = 3.5;
const double LAMBDA = 4.5;

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

typedef std::vector<double> dvec;
typedef std::vector<Legion::Future> state_type;

// -----------------------------------------------------------------------------
// Block-local compute kernels (pure CPU math)
// -----------------------------------------------------------------------------
inline dvec compute_first_block(const dvec &q, double q_r)
{
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
    return dpdt;
}

inline dvec compute_center_block(const dvec &q, double q_l, double q_r)
{
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
    return dpdt;
}

inline dvec compute_last_block(const dvec &q, double q_l)
{
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1.0);
    return dpdt;
}

// -----------------------------------------------------------------------------
// Legion task IDs + task variants
// -----------------------------------------------------------------------------
enum SystemTaskIDs : Legion::TaskID {
    SYSTEM_FIRST_BLOCK_TASK_ID  = 2001,
    SYSTEM_CENTER_BLOCK_TASK_ID = 2002,
    SYSTEM_LAST_BLOCK_TASK_ID   = 2003,
    SYSTEM_SINGLE_BLOCK_TASK_ID = 2004
};

inline dvec system_first_block_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> & /*regions*/,
    Legion::Context /*ctx*/,
    Legion::Runtime * /*runtime*/)
{
    const dvec q       = task->futures[0].get_result<dvec>();
    const dvec q_right = task->futures[1].get_result<dvec>();
    const double q_r   = q_right.empty() ? 0.0 : q_right.front();
    return compute_first_block(q, q_r);
}

inline dvec system_center_block_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> & /*regions*/,
    Legion::Context /*ctx*/,
    Legion::Runtime * /*runtime*/)
{
    const dvec q_left  = task->futures[0].get_result<dvec>();
    const dvec q       = task->futures[1].get_result<dvec>();
    const dvec q_right = task->futures[2].get_result<dvec>();

    const double q_l = q_left.empty()  ? 0.0 : q_left.back();
    const double q_r = q_right.empty() ? 0.0 : q_right.front();

    return compute_center_block(q, q_l, q_r);
}

inline dvec system_last_block_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> & /*regions*/,
    Legion::Context /*ctx*/,
    Legion::Runtime * /*runtime*/)
{
    const dvec q_left = task->futures[0].get_result<dvec>();
    const dvec q      = task->futures[1].get_result<dvec>();
    const double q_l  = q_left.empty() ? 0.0 : q_left.back();
    return compute_last_block(q, q_l);
}

inline dvec system_single_block_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> & /*regions*/,
    Legion::Context /*ctx*/,
    Legion::Runtime * /*runtime*/)
{
    const dvec q = task->futures[0].get_result<dvec>();
    return compute_center_block(q, 0.0, 0.0);
}

inline void preregister_system_tasks()
{
    using namespace Legion;

    {
        TaskVariantRegistrar r(SYSTEM_FIRST_BLOCK_TASK_ID, "system_first_block_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dvec, system_first_block_task>(r, "system_first_block_task");
    }
    {
        TaskVariantRegistrar r(SYSTEM_CENTER_BLOCK_TASK_ID, "system_center_block_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dvec, system_center_block_task>(r, "system_center_block_task");
    }
    {
        TaskVariantRegistrar r(SYSTEM_LAST_BLOCK_TASK_ID, "system_last_block_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dvec, system_last_block_task>(r, "system_last_block_task");
    }
    {
        TaskVariantRegistrar r(SYSTEM_SINGLE_BLOCK_TASK_ID, "system_single_block_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dvec, system_single_block_task>(r, "system_single_block_task");
    }
}

// -----------------------------------------------------------------------------
// Runtime binding helper so osc_chain keeps a 2-argument signature for odeint
// -----------------------------------------------------------------------------
namespace legion_runtime_binding {
inline Legion::Context g_ctx = Legion::Context::NO_CONTEXT;
inline Legion::Runtime *g_runtime = nullptr;

inline void bind(Legion::Context ctx, Legion::Runtime *runtime)
{
    g_ctx = ctx;
    g_runtime = runtime;
}
} // namespace legion_runtime_binding

inline void bind_legion_runtime(Legion::Context ctx, Legion::Runtime *runtime)
{
    legion_runtime_binding::bind(ctx, runtime);
}

// -----------------------------------------------------------------------------
// System assembly
// -----------------------------------------------------------------------------
inline void osc_chain(state_type &q, state_type &dpdt, Legion::Context ctx, Legion::Runtime *runtime)
{
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    using namespace Legion;

    if (N == 1) {
        TaskLauncher l(SYSTEM_SINGLE_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        l.add_future(q[0]);
        dpdt[0] = runtime->execute_task(ctx, l);
        return;
    }

    {
        TaskLauncher l(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        l.add_future(q[0]);
        l.add_future(q[1]);
        dpdt[0] = runtime->execute_task(ctx, l);
    }

    for (size_t i = 1; i < N - 1; ++i) {
        TaskLauncher l(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        l.add_future(q[i - 1]);
        l.add_future(q[i]);
        l.add_future(q[i + 1]);
        dpdt[i] = runtime->execute_task(ctx, l);
    }

    {
        TaskLauncher l(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        l.add_future(q[N - 2]);
        l.add_future(q[N - 1]);
        dpdt[N - 1] = runtime->execute_task(ctx, l);
    }
}

// Backward-compatible wrapper (requires prior bind_legion_runtime call)
inline void osc_chain(state_type &q, state_type &dpdt)
{
    if (legion_runtime_binding::g_runtime == nullptr ||
        legion_runtime_binding::g_ctx == Legion::Context::NO_CONTEXT) {
        throw std::runtime_error(
            "osc_chain called without Legion runtime binding. "
            "Call bind_legion_runtime(ctx, runtime) in your top-level Legion task first.");
    }
    osc_chain(q, dpdt, legion_runtime_binding::g_ctx, legion_runtime_binding::g_runtime);
}

inline void wait_all(state_type &futures)
{
    for (auto &f : futures) f.wait();
}

inline void osc_chain_gb(state_type &q, state_type &dpdt, Legion::Context ctx, Legion::Runtime *runtime)
{
    osc_chain(q, dpdt, ctx, runtime);
    wait_all(dpdt); // global barrier equivalent
}

inline void osc_chain_gb(state_type &q, state_type &dpdt)
{
    osc_chain(q, dpdt);
    wait_all(dpdt); // global barrier equivalent
}

// -----------------------------------------------------------------------------
// Energy diagnostics
// -----------------------------------------------------------------------------
inline double energy(const dvec &q, const dvec &p)
{
    using checked_math::pow;
    using std::abs;

    const size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i]
          +  pow(q[i], KAPPA) / KAPPA
          +  pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
      +  pow(q[N - 1], KAPPA) / KAPPA
      + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;

    return e;
}

template <typename S>
double energy(const S &q_fut, const S &p_fut)
{
    dvec q, p;
    for (size_t i = 0; i < q_fut.size(); ++i) {
        dvec q_block = q_fut[i].template get_result<dvec>();
        dvec p_block = p_fut[i].template get_result<dvec>();
        q.insert(q.end(), q_block.begin(), q_block.end());
        p.insert(p.end(), p_block.begin(), p_block.end());
    }
    return energy(q, p);
}

#endif
