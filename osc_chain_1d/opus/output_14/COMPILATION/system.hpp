// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <legion.h>
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Field and Task IDs shared across the project
enum FieldIDs {
    FID_VAL = 101,
};

enum SystemTaskIDs {
    FIRST_BLOCK_TASK_ID  = 20,
    CENTER_BLOCK_TASK_ID = 21,
    LAST_BLOCK_TASK_ID   = 22,
    ENERGY_TASK_ID       = 23,
};

// Legion state type: a partitioned logical region of doubles
struct state_type {
    LogicalRegion   lr;
    LogicalPartition lp;
    IndexSpace      color_space;
    size_t          num_blocks;  // M
    size_t          block_size;  // G
};

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

// Helper: get the i-th sub-region of a partitioned state
inline LogicalRegion get_subregion(Runtime *runtime,
                                   const state_type &s, size_t i)
{
    return runtime->get_logical_subregion_by_color(
        s.lp, Point<1>(static_cast<coord_t>(i)));
}

// ============================================================
// Task: compute dpdt for the first block (block 0)
// Regions: [0] own q (RO), [1] next q (RO), [2] own dpdt (WD)
// ============================================================
void first_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    qn_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dp_acc(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space()));
    coord_t lo = rect.lo[0], hi = rect.hi[0];

    Rect<1> nr(runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space()));
    double q_r = qn_acc[nr.lo];

    double coupling = -signed_pow(q_acc[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling;
        coupling = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling;
        dp_acc[i] = val;
    }
    dp_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1) + coupling
                 - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: compute dpdt for a center block (blocks 1..M-2)
// Regions: [0] own q (RO), [1] prev q (RO),
//          [2] next q (RO), [3] own dpdt (WD)
// ============================================================
void center_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    qp_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    qn_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dp_acc(regions[3], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space()));
    coord_t lo = rect.lo[0], hi = rect.hi[0];

    Rect<1> pr(runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space()));
    double q_l = qp_acc[pr.hi];

    Rect<1> nr(runtime->get_index_space_domain(ctx,
        regions[2].get_logical_region().get_index_space()));
    double q_r = qn_acc[nr.lo];

    double coupling = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling;
        coupling = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling;
        dp_acc[i] = val;
    }
    dp_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1) + coupling
                 - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: compute dpdt for the last block (block M-1)
// Regions: [0] own q (RO), [1] prev q (RO), [2] own dpdt (WD)
// ============================================================
void last_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    qp_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dp_acc(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space()));
    coord_t lo = rect.lo[0], hi = rect.hi[0];

    Rect<1> pr(runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space()));
    double q_l = qp_acc[pr.hi];

    double coupling = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling;
        coupling = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling;
        dp_acc[i] = val;
    }
    dp_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1) + coupling
                 - signed_pow(q_acc[hi], LAMBDA - 1);
}

// ============================================================
// Launch the oscillator chain system: dpdt = f(q)
// Analogous to HPX osc_chain using dataflow tasks
// ============================================================
void osc_chain(const state_type &q, state_type &dpdt,
               Context ctx, Runtime *runtime)
{
    const size_t M = q.num_blocks;
    assert(M >= 2 && "Need at least 2 blocks for osc_chain");

    // First block (block 0)
    {
        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, 0),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, dpdt, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks (blocks 1..M-2)
    for (size_t i = 1; i < M - 1; ++i)
    {
        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, i),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, i - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, i + 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, dpdt, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block (block M-1)
    {
        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, M - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, q, M - 2),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(runtime, dpdt, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// Global-barrier variant (analogous to HPX osc_chain_gb with wait_all)
void osc_chain_gb(const state_type &q, state_type &dpdt,
                  Context ctx, Runtime *runtime)
{
    osc_chain(q, dpdt, ctx, runtime);
    // Execution fence ensures all launched dpdt tasks complete
    // before any subsequently launched tasks can proceed
    runtime->issue_execution_fence(ctx);
}

// ============================================================
// Energy computation task (reads full q and p regions)
// Regions: [0] q (RO), [1] p (RO)
// ============================================================
double energy_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space()));
    coord_t lo = rect.lo[0], hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double e = 0.5 * pow(abs(q[lo]), LAMBDA) / LAMBDA;
    for (coord_t i = lo; i < hi; ++i)
    {
        double qi  = q[i];
        double pi  = p[i];
        double qi1 = q[i + 1];
        e += 0.5 * pi * pi + pow(qi, KAPPA) / KAPPA
             + pow(abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    double qN = q[hi];
    double pN = p[hi];
    e += 0.5 * pN * pN + pow(qN, KAPPA) / KAPPA
         + 0.5 * pow(abs(qN), LAMBDA) / LAMBDA;
    return e;
}

// Energy launcher: reads full q and p regions, returns scalar energy
double energy(const state_type &q, const state_type &p,
              Context ctx, Runtime *runtime)
{
    TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
    launcher.add_region_requirement(
        RegionRequirement(q.lr, READ_ONLY, EXCLUSIVE, q.lr));
    launcher.region_requirements.back().add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(p.lr, READ_ONLY, EXCLUSIVE, p.lr));
    launcher.region_requirements.back().add_field(FID_VAL);
    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

// ============================================================
// Task registration (call before Runtime::start)
// ============================================================
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar r(FIRST_BLOCK_TASK_ID, "first_block");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<first_block_task>(
            r, "first_block");
    }
    {
        TaskVariantRegistrar r(CENTER_BLOCK_TASK_ID, "center_block");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<center_block_task>(
            r, "center_block");
    }
    {
        TaskVariantRegistrar r(LAST_BLOCK_TASK_ID, "last_block");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<last_block_task>(
            r, "last_block");
    }
    {
        TaskVariantRegistrar r(ENERGY_TASK_ID, "energy_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, energy_task_impl>(
            r, "energy_task");
    }
}

#endif
