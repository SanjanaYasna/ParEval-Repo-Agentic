// Copyright 2013 Mario Mulansky
// Translated to Legion execution model
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <cassert>

#include "legion.h"

const double KAPPA = 3.5;
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
    const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow(x, k) * s;
}

typedef std::vector<double> dvec;

// Each block is represented by a LogicalRegion that stores one field (FID_VALUE)
typedef std::vector<Legion::LogicalRegion> state_type;

enum SystemTaskIDs : Legion::TaskID {
    SYSTEM_BLOCK_TASK_ID = 1001
};

enum SystemFieldIDs : Legion::FieldID {
    FID_VALUE = 1
};

struct SystemTaskArgs {
    int has_left;
    int has_right;
};

// Optional global runtime/context hooks to keep call sites simple.
inline Legion::Runtime*& system_runtime()
{
    static Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& system_context()
{
    static Legion::Context ctx = Legion::Context();
    return ctx;
}

inline bool& system_ctx_set()
{
    static bool is_set = false;
    return is_set;
}

inline void set_system_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    system_runtime() = runtime;
    system_context() = ctx;
    system_ctx_set() = true;
}

inline void system_block_task(const Legion::Task* task,
                              const std::vector<Legion::PhysicalRegion>& regions,
                              Legion::Context ctx,
                              Legion::Runtime* runtime)
{
    assert(task->arglen == sizeof(SystemTaskArgs));
    const auto* args = static_cast<const SystemTaskArgs*>(task->args);

    // regions[0] -> q(current), regions[1] -> dpdt(current), [2]/[3] optional neighbors
    Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(regions[0], FID_VALUE);
    Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VALUE);

    const Legion::LogicalRegion q_lr = task->regions[0].region;
    Legion::Rect<1> q_rect = runtime->get_index_space_domain(ctx, q_lr.get_index_space());
    if (q_rect.hi[0] < q_rect.lo[0]) return; // empty block

    double q_l = 0.0;
    double q_r = 0.0;
    unsigned idx = 2;

    if (args->has_left) {
        Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_left_acc(regions[idx], FID_VALUE);
        const Legion::LogicalRegion q_left_lr = task->regions[idx].region;
        Legion::Rect<1> left_rect = runtime->get_index_space_domain(ctx, q_left_lr.get_index_space());
        q_l = q_left_acc[Legion::Point<1>(left_rect.hi[0])];
        idx++;
    }

    if (args->has_right) {
        Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_right_acc(regions[idx], FID_VALUE);
        const Legion::LogicalRegion q_right_lr = task->regions[idx].region;
        Legion::Rect<1> right_rect = runtime->get_index_space_domain(ctx, q_right_lr.get_index_space());
        q_r = q_right_acc[Legion::Point<1>(right_rect.lo[0])];
    }

    const Legion::coord_t lo = q_rect.lo[0];
    const Legion::coord_t hi = q_rect.hi[0];

    double coupling_lr = -signed_pow(q_acc[Legion::Point<1>(lo)] - q_l, LAMBDA - 1.0);

    for (Legion::coord_t p = lo; p < hi; ++p) {
        const double qi = q_acc[Legion::Point<1>(p)];
        const double qip1 = q_acc[Legion::Point<1>(p + 1)];
        dpdt_acc[Legion::Point<1>(p)] = -signed_pow(qi, KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(qi - qip1, LAMBDA - 1.0);
        dpdt_acc[Legion::Point<1>(p)] -= coupling_lr;
    }

    const double q_last = q_acc[Legion::Point<1>(hi)];
    dpdt_acc[Legion::Point<1>(hi)] =
        -signed_pow(q_last, KAPPA - 1.0) + coupling_lr
        - signed_pow(q_last - q_r, LAMBDA - 1.0);
}

inline void preregister_system_tasks()
{
    static bool done = false;
    if (done) return;
    done = true;

    Legion::TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<system_block_task>(registrar, "system_block_task");
}

inline void osc_chain(state_type& q, state_type& dpdt, Legion::Runtime* runtime, Legion::Context ctx)
{
    assert(q.size() == dpdt.size());
    const std::size_t nblocks = q.size();
    if (nblocks == 0) return;

    for (std::size_t i = 0; i < nblocks; ++i) {
        SystemTaskArgs args{ (i > 0) ? 1 : 0, (i + 1 < nblocks) ? 1 : 0 };
        Legion::TaskLauncher launcher(SYSTEM_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));

        launcher.add_region_requirement(
            Legion::RegionRequirement(q[i], Legion::READ_ONLY, Legion::EXCLUSIVE, q[i]));
        launcher.add_field(0, FID_VALUE);

        launcher.add_region_requirement(
            Legion::RegionRequirement(dpdt[i], Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt[i]));
        launcher.add_field(1, FID_VALUE);

        if (i > 0) {
            launcher.add_region_requirement(
                Legion::RegionRequirement(q[i - 1], Legion::READ_ONLY, Legion::EXCLUSIVE, q[i - 1]));
            launcher.add_field(2, FID_VALUE);
        }
        if (i + 1 < nblocks) {
            const unsigned rr_idx = (i > 0) ? 3 : 2;
            launcher.add_region_requirement(
                Legion::RegionRequirement(q[i + 1], Legion::READ_ONLY, Legion::EXCLUSIVE, q[i + 1]));
            launcher.add_field(rr_idx, FID_VALUE);
        }

        runtime->execute_task(ctx, launcher);
    }
}

// Wrapper retaining original-style signature (requires prior set_system_legion_context call)
inline void osc_chain(state_type& q, state_type& dpdt)
{
    if (!system_ctx_set()) {
        throw std::runtime_error("System Legion context not set. Call set_system_legion_context first.");
    }
    osc_chain(q, dpdt, system_runtime(), system_context());
}

inline void osc_chain_gb(state_type& q, state_type& dpdt, Legion::Runtime* runtime, Legion::Context ctx)
{
    osc_chain(q, dpdt, runtime, ctx);
    Legion::Future f = runtime->issue_execution_fence(ctx);
    f.get_void_result();
}

// Wrapper retaining original-style signature
inline void osc_chain_gb(state_type& q, state_type& dpdt)
{
    if (!system_ctx_set()) {
        throw std::runtime_error("System Legion context not set. Call set_system_legion_context first.");
    }
    osc_chain_gb(q, dpdt, system_runtime(), system_context());
}

inline double energy(const dvec& q, const dvec& p)
{
    using checked_math::pow;
    using std::abs;
    const std::size_t N = q.size();

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i]
           + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

inline void append_region_to_host_vector(const Legion::LogicalRegion& lr,
                                         dvec& out,
                                         Legion::Runtime* runtime,
                                         Legion::Context ctx)
{
    Legion::RegionRequirement req(lr, Legion::READ_ONLY, Legion::EXCLUSIVE, lr);
    req.add_field(FID_VALUE);

    Legion::InlineLauncher launcher(req);
    Legion::PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    Legion::FieldAccessor<Legion::READ_ONLY, double, 1> acc(pr, FID_VALUE);
    Legion::Rect<1> rect = runtime->get_index_space_domain(ctx, lr.get_index_space());
    if (rect.hi[0] >= rect.lo[0]) {
        out.reserve(out.size() + static_cast<std::size_t>(rect.hi[0] - rect.lo[0] + 1));
        for (Legion::coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
            out.push_back(acc[Legion::Point<1>(i)]);
        }
    }

    runtime->unmap_region(ctx, pr);
}

template <typename S>
double energy(const S& q_regions, const S& p_regions, Legion::Runtime* runtime, Legion::Context ctx)
{
    dvec q, p;
    for (std::size_t i = 0; i < q_regions.size(); ++i) {
        append_region_to_host_vector(q_regions[i], q, runtime, ctx);
        append_region_to_host_vector(p_regions[i], p, runtime, ctx);
    }
    return energy(q, p);
}

// Wrapper retaining original-style call shape
inline double energy(const state_type& q_regions, const state_type& p_regions)
{
    if (!system_ctx_set()) {
        throw std::runtime_error("System Legion context not set. Call set_system_legion_context first.");
    }
    return energy(q_regions, p_regions, system_runtime(), system_context());
}

#endif
