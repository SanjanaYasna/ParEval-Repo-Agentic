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

enum SystemTaskIDs {
    TASK_FIRST_BLOCK = 10,
    TASK_CENTER_BLOCK = 11,
    TASK_LAST_BLOCK = 12,
};

enum SystemFieldIDs {
    FID_VAL = 101,
};

// Legion state type: a partitioned logical region
struct state_type {
    LogicalRegion lr;
    LogicalPartition lp;
    size_t num_blocks;  // M
    size_t block_size;  // G
    size_t total_size;  // N
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
    return checked_math::pow(x, k) * sign(x);
}

// ---- Task: first block (no left neighbor) ----
// Region 0: q block 0       (RO)
// Region 1: q block 1       (RO, need first element as q_r)
// Region 2: dpdt block 0    (WD)
inline void system_first_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

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

// ---- Task: center block (has both neighbors) ----
// Region 0: q block i       (RO)
// Region 1: q block i-1     (RO, need last element as q_l)
// Region 2: q block i+1     (RO, need first element as q_r)
// Region 3: dpdt block i    (WD)
inline void system_center_block_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

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

// ---- Task: last block (no right neighbor) ----
// Region 0: q block M-1     (RO)
// Region 1: q block M-2     (RO, need last element as q_l)
// Region 2: dpdt block M-1  (WD)
inline void system_last_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

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

// ---- Launch all block tasks (replaces HPX dataflow graph) ----
inline void osc_chain(const state_type &q, const state_type &dpdt,
                       Context ctx, Runtime *runtime)
{
    const size_t M = q.num_blocks;
    assert(M >= 2);

    // Retrieve logical subregions for each block
    std::vector<LogicalRegion> q_subs(M), dpdt_subs(M);
    for (size_t i = 0; i < M; ++i)
    {
        q_subs[i] = runtime->get_logical_subregion_by_color(ctx, q.lp,
                        DomainPoint(Point<1>((coord_t)i)));
        dpdt_subs[i] = runtime->get_logical_subregion_by_color(ctx, dpdt.lp,
                        DomainPoint(Point<1>((coord_t)i)));
    }

    // First block
    {
        TaskLauncher launcher(TASK_FIRST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_subs[0], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q_subs[1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_subs[0], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        TaskLauncher launcher(TASK_CENTER_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_subs[i], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q_subs[i - 1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q_subs[i + 1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_subs[i], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        TaskLauncher launcher(TASK_LAST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_subs[M - 1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q_subs[M - 2], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_subs[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ---- Energy computation on raw vectors (same as original) ----
inline double energy(const std::vector<double> &q, const std::vector<double> &p)
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

// ---- Energy computation on Legion state (inline-mapped) ----
inline double energy(const state_type &q_state, const state_type &p_state,
                     Context ctx, Runtime *runtime)
{
    using checked_math::pow;
    using std::abs;

    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);

    InlineLauncher q_il(q_req);
    InlineLauncher p_il(p_req);

    PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

    const coord_t N = (coord_t)q_state.total_size;

    double en = 0.5 * pow(abs((double)q_acc[0]), LAMBDA) / LAMBDA;
    for (coord_t i = 0; i < N - 1; ++i)
    {
        double qi  = q_acc[i];
        double qi1 = q_acc[i + 1];
        double pi  = p_acc[i];
        en += 0.5 * pi * pi + pow(qi, KAPPA) / KAPPA
            + pow(abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    {
        double qN = q_acc[N - 1];
        double pN = p_acc[N - 1];
        en += 0.5 * pN * pN + pow(qN, KAPPA) / KAPPA
            + 0.5 * pow(abs(qN), LAMBDA) / LAMBDA;
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return en;
}

// ---- Register all system tasks (call before Runtime::start) ----
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(TASK_FIRST_BLOCK, "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_CENTER_BLOCK, "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_LAST_BLOCK, "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif
