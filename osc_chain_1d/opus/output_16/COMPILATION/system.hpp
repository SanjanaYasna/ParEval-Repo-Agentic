// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include "legion.h"
#include <vector>
#include <memory>
#include <cmath>
#include <cassert>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

typedef std::vector<double> dvec;

// Field ID shared across the application
enum SystemFieldID {
    FID_VAL = 100,
};

// Task IDs for system computation
enum SystemTaskID {
    TASK_EXTRACT_FIRST     = 20,
    TASK_EXTRACT_LAST      = 21,
    TASK_SYSTEM_FIRST_BLK  = 22,
    TASK_SYSTEM_CENTER_BLK = 23,
    TASK_SYSTEM_LAST_BLK   = 24,
};

// State representation: a partitioned 1D logical region
struct state_partition {
    LogicalRegion    lr;  // Full logical region (parent)
    LogicalPartition lp;  // Equal partition into M sub-regions
    size_t           M;   // Number of partitions
};

// ---------------------------------------------------------------------------
// Math utilities (unchanged from HPX version)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper: get sub-region by integer color
// ---------------------------------------------------------------------------

static inline LogicalRegion get_subregion(Runtime *runtime, Context ctx,
                                          LogicalPartition lp, size_t color)
{
    return runtime->get_logical_subregion_by_color(ctx, lp,
        DomainPoint(Point<1>((coord_t)color)));
}

// ---------------------------------------------------------------------------
// Leaf task: extract first element of a sub-region  (returns double)
// ---------------------------------------------------------------------------

inline double extract_first_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    return acc[Point<1>(rect.lo[0])];
}

// ---------------------------------------------------------------------------
// Leaf task: extract last element of a sub-region  (returns double)
// ---------------------------------------------------------------------------

inline double extract_last_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    return acc[Point<1>(rect.hi[0])];
}

// ---------------------------------------------------------------------------
// Leaf task: compute forces for the FIRST block (left boundary condition)
//   Region 0: q   (READ_ONLY)
//   Region 1: dpdt (WRITE_DISCARD)
//   Future 0: q_r  (first element of the next block)
// ---------------------------------------------------------------------------

inline void system_first_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    double q_r = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    coord_t lo = rect.lo[0];
    size_t  N  = (size_t)(rect.hi[0] - lo + 1);

    // Copy q into a local buffer for efficient sequential access
    dvec q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = acc_q[Point<1>(lo + (coord_t)i)];

    // Compute forces — left boundary: coupling with implicit zero
    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(lo + (coord_t)(N - 1))] =
        -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr
        - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// ---------------------------------------------------------------------------
// Leaf task: compute forces for a CENTER block
//   Region 0: q    (READ_ONLY)
//   Region 1: dpdt (WRITE_DISCARD)
//   Future 0: q_l  (last element of the previous block)
//   Future 1: q_r  (first element of the next block)
// ---------------------------------------------------------------------------

inline void system_center_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();
    double q_r = task->futures[1].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    coord_t lo = rect.lo[0];
    size_t  N  = (size_t)(rect.hi[0] - lo + 1);

    dvec q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = acc_q[Point<1>(lo + (coord_t)i)];

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(lo + (coord_t)(N - 1))] =
        -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr
        - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// ---------------------------------------------------------------------------
// Leaf task: compute forces for the LAST block (right boundary condition)
//   Region 0: q    (READ_ONLY)
//   Region 1: dpdt (WRITE_DISCARD)
//   Future 0: q_l  (last element of the previous block)
// ---------------------------------------------------------------------------

inline void system_last_block_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    coord_t lo = rect.lo[0];
    size_t  N  = (size_t)(rect.hi[0] - lo + 1);

    dvec q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = acc_q[Point<1>(lo + (coord_t)i)];

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(lo + (coord_t)(N - 1))] =
        -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr
        - signed_pow(q[N - 1], LAMBDA - 1);
}

// ---------------------------------------------------------------------------
// osc_chain  —  launch force-computation tasks (non-blocking)
//
// Mirrors the HPX dataflow graph:
//   1. Extract boundary scalars from each q sub-region  → Future<double>
//   2. Launch block tasks that consume those futures and write to dpdt
//
// Legion's dependency tracking ensures correct ordering; all block tasks
// on disjoint dpdt sub-regions may execute in parallel.
// ---------------------------------------------------------------------------

inline void osc_chain(const state_partition &q, const state_partition &dpdt,
                      Context ctx, Runtime *runtime)
{
    const size_t M = q.M;
    assert(M >= 2 && "Oscillator chain requires at least 2 blocks");

    // ------ Phase 1: extract boundary values as futures ------------------
    std::vector<Future> first_elem(M);  // first element of each q block
    std::vector<Future> last_elem(M);   // last element of each q block

    for (size_t i = 0; i < M; i++)
    {
        LogicalRegion q_sub = get_subregion(runtime, ctx, q.lp, i);

        // Extract first element
        {
            TaskLauncher launcher(TASK_EXTRACT_FIRST, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.add_field(0, FID_VAL);
            first_elem[i] = runtime->execute_task(ctx, launcher);
        }
        // Extract last element
        {
            TaskLauncher launcher(TASK_EXTRACT_LAST, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.add_field(0, FID_VAL);
            last_elem[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // ------ Phase 2: launch block computation tasks ----------------------

    // First block  (left boundary)
    {
        LogicalRegion q_sub    = get_subregion(runtime, ctx, q.lp, 0);
        LogicalRegion dpdt_sub = get_subregion(runtime, ctx, dpdt.lp, 0);

        TaskLauncher launcher(TASK_SYSTEM_FIRST_BLK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(1, FID_VAL);
        launcher.add_future(first_elem[1]);          // q_r
        runtime->execute_task(ctx, launcher);
    }

    // Middle blocks
    for (size_t i = 1; i < M - 1; i++)
    {
        LogicalRegion q_sub    = get_subregion(runtime, ctx, q.lp, i);
        LogicalRegion dpdt_sub = get_subregion(runtime, ctx, dpdt.lp, i);

        TaskLauncher launcher(TASK_SYSTEM_CENTER_BLK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(1, FID_VAL);
        launcher.add_future(last_elem[i - 1]);       // q_l
        launcher.add_future(first_elem[i + 1]);      // q_r
        runtime->execute_task(ctx, launcher);
    }

    // Last block  (right boundary)
    {
        LogicalRegion q_sub    = get_subregion(runtime, ctx, q.lp, M - 1);
        LogicalRegion dpdt_sub = get_subregion(runtime, ctx, dpdt.lp, M - 1);

        TaskLauncher launcher(TASK_SYSTEM_LAST_BLK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(1, FID_VAL);
        launcher.add_future(last_elem[M - 2]);       // q_l
        runtime->execute_task(ctx, launcher);
    }
}

// ---------------------------------------------------------------------------
// osc_chain_gb  —  global-barrier version (waits for all dpdt writes)
// ---------------------------------------------------------------------------

inline void osc_chain_gb(const state_partition &q, const state_partition &dpdt,
                         Context ctx, Runtime *runtime)
{
    osc_chain(q, dpdt, ctx, runtime);
    // Execution fence: all tasks launched before this point must complete
    // before any task launched after this point may begin.
    runtime->issue_execution_fence(ctx);
}

// ---------------------------------------------------------------------------
// Energy computation  (sequential, on gathered data)
// ---------------------------------------------------------------------------

inline double energy(const dvec &q, const dvec &p)
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double en = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i)
    {
        en += 0.5 * p[i] * p[i]
            + pow(q[i], KAPPA) / KAPPA
            + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    en += 0.5 * p[N - 1] * p[N - 1]
        + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return en;
}

// Overload that reads from Legion logical regions (inline-mapped).
// The inline mapping automatically waits for any prior writes to complete.
inline double energy(const state_partition &q_sp, const state_partition &p_sp,
                     Context ctx, Runtime *runtime)
{
    // Inline-map the full q region
    RegionRequirement q_req(q_sp.lr, READ_ONLY, EXCLUSIVE, q_sp.lr);
    q_req.add_field(FID_VAL);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_req);
    q_pr.wait_until_valid();

    // Inline-map the full p region
    RegionRequirement p_req(p_sp.lr, READ_ONLY, EXCLUSIVE, p_sp.lr);
    p_req.add_field(FID_VAL);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_req);
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> acc_q(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_p(p_pr, FID_VAL);

    Domain dom = runtime->get_index_space_domain(q_sp.lr.get_index_space());
    Rect<1> rect = dom;
    size_t N = (size_t)(rect.hi[0] - rect.lo[0] + 1);

    // Gather into flat vectors (mirrors the HPX version)
    dvec q_vec(N), p_vec(N);
    for (size_t i = 0; i < N; i++)
    {
        q_vec[i] = acc_q[Point<1>(rect.lo[0] + (coord_t)i)];
        p_vec[i] = acc_p[Point<1>(rect.lo[0] + (coord_t)i)];
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return energy(q_vec, p_vec);
}

// ---------------------------------------------------------------------------
// Task registration  —  call once before Runtime::start()
// ---------------------------------------------------------------------------

inline void register_system_tasks()
{
    // extract_first  (leaf, returns double)
    {
        TaskVariantRegistrar registrar(TASK_EXTRACT_FIRST, "extract_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, extract_first_task>(
            registrar, "extract_first");
    }
    // extract_last  (leaf, returns double)
    {
        TaskVariantRegistrar registrar(TASK_EXTRACT_LAST, "extract_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, extract_last_task>(
            registrar, "extract_last");
    }
    // system_first_block  (leaf, void)
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_FIRST_BLK, "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    // system_center_block  (leaf, void)
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_CENTER_BLK, "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    // system_last_block  (leaf, void)
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_LAST_BLK, "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif
