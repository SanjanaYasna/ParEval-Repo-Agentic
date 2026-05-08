// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include "legion.h"
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum FieldIDs {
    FID_VAL = 101,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    FORCE_FIRST_BLOCK_TASK_ID,
    FORCE_CENTER_BLOCK_TASK_ID,
    FORCE_LAST_BLOCK_TASK_ID,
    FORCE_SINGLE_BLOCK_TASK_ID,
    ENERGY_TASK_ID,
    SCALE_SUM2_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
};

namespace checked_math {
    inline double pow( double x , double y )
    {
        if( x==0.0 )
            return 0.0;
        using std::pow;
        using std::abs;
        return pow( abs(x) , y );
    }
}

inline double signed_pow( double x , double k )
{
    using boost::math::sign;
    using std::abs;
    return checked_math::pow( x , k ) * sign(x);
}

// State type: a partitioned 1D logical region of doubles
struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    IndexSpace color_space;
    size_t N; // total number of elements
    size_t M; // number of blocks (dataflows)
    size_t G; // elements per block
};

// Helper to retrieve a subregion by color index
inline LogicalRegion get_subregion(Runtime *runtime, Context ctx,
                                   LogicalPartition lp, int color)
{
    return runtime->get_logical_subregion_by_color(ctx, lp, Point<1>(color));
}

// ===================== Force task implementations =====================

// First block force computation
// Region 0: q this block       (READ_ONLY)
// Region 1: q right neighbor   (READ_ONLY)
// Region 2: dpdt this block    (WRITE_DISCARD)
void force_first_block_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // First element of right neighbor block
    double q_r = q_right_acc[right_rect.lo];

    double coupling_lr = -signed_pow(q_acc[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// Center block force computation
// Region 0: q this block       (READ_ONLY)
// Region 1: q left neighbor    (READ_ONLY)
// Region 2: q right neighbor   (READ_ONLY)
// Region 3: dpdt this block    (WRITE_DISCARD)
void force_center_block_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Last element of left neighbor, first element of right neighbor
    double q_l = q_left_acc[left_rect.hi];
    double q_r = q_right_acc[right_rect.lo];

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// Last block force computation
// Region 0: q this block       (READ_ONLY)
// Region 1: q left neighbor    (READ_ONLY)
// Region 2: dpdt this block    (WRITE_DISCARD)
void force_last_block_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Last element of left neighbor
    double q_l = q_left_acc[left_rect.hi];

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi], LAMBDA - 1);
}

// Single block force computation (M == 1)
// Region 0: q  (READ_ONLY)
// Region 1: dpdt (WRITE_DISCARD)
void force_single_block_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    double coupling_lr = -signed_pow(q_acc[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi], LAMBDA - 1);
}

// ===================== System function =====================
// Launches per-block force tasks (asynchronous).
// Legion's dependence analysis orders these with respect to
// subsequent tasks that touch the same regions.

void osc_chain(state_type &q, state_type &dpdt,
               Runtime *runtime, Context ctx)
{
    const size_t M = q.M;

    // Cache subregions
    std::vector<LogicalRegion> q_sub(M), dpdt_sub(M);
    for (size_t i = 0; i < M; ++i)
    {
        q_sub[i]    = get_subregion(runtime, ctx, q.partition, i);
        dpdt_sub[i] = get_subregion(runtime, ctx, dpdt.partition, i);
    }

    if (M == 1)
    {
        TaskLauncher launcher(FORCE_SINGLE_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        unsigned idx;
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[0], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(dpdt_sub[0], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(idx, FID_VAL);
        runtime->execute_task(ctx, launcher);
        return;
    }

    // First block
    {
        TaskLauncher launcher(FORCE_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        unsigned idx;
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[0], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[1], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(dpdt_sub[0], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(idx, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        TaskLauncher launcher(FORCE_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        unsigned idx;
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[i], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[i - 1], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[i + 1], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(dpdt_sub[i], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(idx, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        TaskLauncher launcher(FORCE_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        unsigned idx;
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[M - 1], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[M - 2], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(dpdt_sub[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(idx, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// Global-barrier variant: waits for all force tasks to complete
void osc_chain_gb(state_type &q, state_type &dpdt,
                  Runtime *runtime, Context ctx)
{
    const size_t M = q.M;
    std::vector<Future> futures;

    std::vector<LogicalRegion> q_sub(M), dpdt_sub(M);
    for (size_t i = 0; i < M; ++i)
    {
        q_sub[i]    = get_subregion(runtime, ctx, q.partition, i);
        dpdt_sub[i] = get_subregion(runtime, ctx, dpdt.partition, i);
    }

    if (M == 1)
    {
        TaskLauncher launcher(FORCE_SINGLE_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        unsigned idx;
        idx = launcher.add_region_requirement(
            RegionRequirement(q_sub[0], READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(idx, FID_VAL);
        idx = launcher.add_region_requirement(
            RegionRequirement(dpdt_sub[0], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(idx, FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }
    else
    {
        // First block
        {
            TaskLauncher launcher(FORCE_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
            unsigned idx;
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[0], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[1], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(dpdt_sub[0], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
            launcher.add_field(idx, FID_VAL);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }
        // Center blocks
        for (size_t i = 1; i < M - 1; ++i)
        {
            TaskLauncher launcher(FORCE_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
            unsigned idx;
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[i], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[i - 1], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[i + 1], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(dpdt_sub[i], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
            launcher.add_field(idx, FID_VAL);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }
        // Last block
        {
            TaskLauncher launcher(FORCE_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
            unsigned idx;
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[M - 1], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(q_sub[M - 2], READ_ONLY, EXCLUSIVE, q.region));
            launcher.add_field(idx, FID_VAL);
            idx = launcher.add_region_requirement(
                RegionRequirement(dpdt_sub[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.region));
            launcher.add_field(idx, FID_VAL);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }
    }

    // Global barrier: wait for all force tasks
    for (auto &f : futures)
        f.get_void_result();
}

// ===================== Energy computation =====================

// Energy leaf task: reads full q and p regions, returns total energy
double energy_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double en = 0.5 * pow( abs(q_acc[lo]) , LAMBDA ) / LAMBDA;
    for (coord_t i = lo; i < hi; ++i)
    {
        en += 0.5 * p_acc[i] * p_acc[i]
            + pow( q_acc[i] , KAPPA ) / KAPPA
            + pow( abs(q_acc[i] - q_acc[i + 1]) , LAMBDA ) / LAMBDA;
    }
    en += 0.5 * p_acc[hi] * p_acc[hi]
        + pow( q_acc[hi] , KAPPA ) / KAPPA
        + 0.5 * pow( abs(q_acc[hi]) , LAMBDA ) / LAMBDA;

    return en;
}

// Launch the energy task and block for the result
double energy(const state_type &q, const state_type &p,
              Runtime *runtime, Context ctx)
{
    TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
    unsigned idx;
    idx = launcher.add_region_requirement(
        RegionRequirement(q.region, READ_ONLY, EXCLUSIVE, q.region));
    launcher.add_field(idx, FID_VAL);
    idx = launcher.add_region_requirement(
        RegionRequirement(p.region, READ_ONLY, EXCLUSIVE, p.region));
    launcher.add_field(idx, FID_VAL);
    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

// ===================== Task registration =====================

void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(FORCE_FIRST_BLOCK_TASK_ID,
                                       "force_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<force_first_block_task>(
            registrar, "force_first_block");
    }
    {
        TaskVariantRegistrar registrar(FORCE_CENTER_BLOCK_TASK_ID,
                                       "force_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<force_center_block_task>(
            registrar, "force_center_block");
    }
    {
        TaskVariantRegistrar registrar(FORCE_LAST_BLOCK_TASK_ID,
                                       "force_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<force_last_block_task>(
            registrar, "force_last_block");
    }
    {
        TaskVariantRegistrar registrar(FORCE_SINGLE_BLOCK_TASK_ID,
                                       "force_single_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<force_single_block_task>(
            registrar, "force_single_block");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, energy_task_impl>(
            registrar, "energy_task");
    }
}

#endif // SYSTEM_HPP
