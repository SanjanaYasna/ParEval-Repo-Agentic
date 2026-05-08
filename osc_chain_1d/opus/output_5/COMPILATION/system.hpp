// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cassert>

#include "legion.h"
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Common task and field IDs shared across the project
#ifndef COMMON_IDS_DEFINED
#define COMMON_IDS_DEFINED
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    SCALE_SUM2_TASK_ID,
};

enum FieldIDs {
    FID_VAL = 0,
};

struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    size_t num_parts;   // M
    size_t block_size;  // G
};
#endif // COMMON_IDS_DEFINED

typedef std::vector<double> dvec;

namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0)
            return 0.0;
        using std::pow;
        using std::abs;
        return pow(abs(x), y);
    }
}

inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    using std::abs;
    return checked_math::pow(x, k) * sign(x);
}

// ============================================================
// Task: compute dpdt for the first block
// Regions: [0] q_block (RO), [1] q_next_block (RO), [2] dpdt_block (WD)
// ============================================================
inline void system_first_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> next_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // First element of next block (coupling neighbor)
    double q_r = q_next[next_rect.lo];

    double coupling_lr = -signed_pow(q[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt[i] = val;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: compute dpdt for a center block
// Regions: [0] q_block (RO), [1] q_prev_block (RO),
//          [2] q_next_block (RO), [3] dpdt_block (WD)
// ============================================================
inline void system_center_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> prev_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));
    Rect<1> next_rect(runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Last element of previous block, first element of next block
    double q_l = q_prev[prev_rect.hi];
    double q_r = q_next[next_rect.lo];

    double coupling_lr = -signed_pow(q[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt[i] = val;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: compute dpdt for the last block
// Regions: [0] q_block (RO), [1] q_prev_block (RO), [2] dpdt_block (WD)
// ============================================================
inline void system_last_block_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> prev_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Last element of previous block
    double q_l = q_prev[prev_rect.hi];

    double coupling_lr = -signed_pow(q[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt[i] = val;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi], LAMBDA - 1);
}

// ============================================================
// System function: launches Legion tasks (non-blocking).
// Legion runtime resolves dependencies via region requirements.
// ============================================================
inline void osc_chain(state_type &q, state_type &dpdt,
                      Runtime *runtime, Context ctx)
{
    const size_t M = q.num_parts;

    // Helper to look up a sub-region by partition color
    auto sub = [&](const state_type &s, size_t color) -> LogicalRegion {
        return runtime->get_logical_subregion_by_color(ctx,
            s.partition, DomainPoint(Point<1>((coord_t)color)));
    };

    // First block
    {
        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(sub(q, 0), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(q, 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(dpdt, 0), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Middle blocks
    for (size_t i = 1; i < M - 1; i++)
    {
        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(sub(q, i), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(q, i - 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(q, i + 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[2].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(dpdt, i), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements[3].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    if (M > 1)
    {
        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(sub(q, M - 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(q, M - 2), READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(sub(dpdt, M - 1), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// Version with global barrier (equivalent to HPX wait_all)
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                         Runtime *runtime, Context ctx)
{
    osc_chain(q, dpdt, runtime, ctx);
    // Execution fence: all previously launched tasks in this context
    // must complete before any subsequently launched tasks begin.
    Future fence = runtime->issue_execution_fence(ctx);
    fence.get_void_result();
}

// ============================================================
// Energy computation (scalar version on plain vectors)
// ============================================================
inline double energy(const dvec &q, const dvec &p)
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double en = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i)
    {
        en += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
            + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    en += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return en;
}

// ============================================================
// Energy computation on Legion state_type (inline-maps regions)
// ============================================================
inline double energy(const state_type &q_state, const state_type &p_state,
                     Runtime *runtime, Context ctx)
{
    // Inline-map both full regions for reading.
    // The runtime will automatically wait for any outstanding
    // writes (e.g. from osc_chain tasks) to complete first.
    RegionRequirement q_req(q_state.region, READ_ONLY, EXCLUSIVE, q_state.region);
    q_req.add_field(FID_VAL);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_req);
    q_phys.wait_until_valid();

    RegionRequirement p_req(p_state.region, READ_ONLY, EXCLUSIVE, p_state.region);
    p_req.add_field(FID_VAL);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_req);
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        q_state.region.get_index_space()));
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    // Gather into contiguous vectors and reuse scalar energy()
    dvec q_vec(N), p_vec(N);
    for (coord_t i = lo; i <= hi; ++i)
    {
        q_vec[(size_t)(i - lo)] = q_acc[i];
        p_vec[(size_t)(i - lo)] = p_acc[i];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q_vec, p_vec);
}

// ============================================================
// Task registration (call from main before Runtime::start)
// ============================================================
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(SYSTEM_FIRST_BLOCK_TASK_ID,
                                       "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_CENTER_BLOCK_TASK_ID,
                                       "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_LAST_BLOCK_TASK_ID,
                                       "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif // SYSTEM_HPP
