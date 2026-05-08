// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <legion.h>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum FieldIDs {
    FID_VAL = 101,
};

enum SystemTaskIDs {
    TASK_SYSTEM_FIRST_BLOCK = 20,
    TASK_SYSTEM_CENTER_BLOCK = 21,
    TASK_SYSTEM_LAST_BLOCK = 22,
};

struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    size_t num_parts;
    size_t block_size;

    inline LogicalRegion get_subregion(Runtime *runtime, Context ctx,
                                       size_t idx) const
    {
        return runtime->get_logical_subregion_by_color(
            partition, DomainPoint(Point<1>(static_cast<coord_t>(idx))));
    }
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

/* ------------------------------------------------------------------ */
/*  Task implementations                                              */
/* ------------------------------------------------------------------ */

inline void system_first_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    // Region 0 : q  block 0       (RO)
    // Region 1 : q  block 1       (RO) – only first element needed
    // Region 2 : dpdt block 0     (WD)
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    q_next_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_next_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_r = q_next_acc[q_next_rect.lo];

    double coupling_lr = -signed_pow(q_acc[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

inline void system_center_block_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime *runtime)
{
    // Region 0 : q  block i       (RO)
    // Region 1 : q  block i-1     (RO) – only last element needed
    // Region 2 : q  block i+1     (RO) – only first element needed
    // Region 3 : dpdt block i     (WD)
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    q_next_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_prev_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());
    Rect<1> q_next_rect = runtime->get_index_space_domain(
        regions[2].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_l = q_prev_acc[q_prev_rect.hi];
    double q_r = q_next_acc[q_next_rect.lo];

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

inline void system_last_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    // Region 0 : q  block M-1     (RO)
    // Region 1 : q  block M-2     (RO) – only last element needed
    // Region 2 : dpdt block M-1   (WD)
    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_prev_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_l = q_prev_acc[q_prev_rect.hi];

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi], LAMBDA - 1);
}

/* ------------------------------------------------------------------ */
/*  osc_chain  –  launch one task per block (non-blocking)            */
/* ------------------------------------------------------------------ */

inline void osc_chain(state_type &q, state_type &dpdt,
                       Runtime *runtime, Context ctx)
{
    const size_t M = q.num_parts;

    // First block
    {
        TaskLauncher launcher(TASK_SYSTEM_FIRST_BLOCK, TaskArgument(NULL, 0));

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, 0),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(TASK_SYSTEM_CENTER_BLOCK, TaskArgument(NULL, 0));

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i - 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i + 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }

    // Last block
    if (M > 1) {
        TaskLauncher launcher(TASK_SYSTEM_LAST_BLOCK, TaskArgument(NULL, 0));

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, M - 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, M - 2),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }
}

/* ------------------------------------------------------------------ */
/*  osc_chain_gb  –  same as osc_chain but with a global barrier      */
/* ------------------------------------------------------------------ */

inline void osc_chain_gb(state_type &q, state_type &dpdt,
                          Runtime *runtime, Context ctx)
{
    const size_t M = q.num_parts;
    std::vector<Future> futures;

    // First block
    {
        TaskLauncher launcher(TASK_SYSTEM_FIRST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, 0),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(TASK_SYSTEM_CENTER_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i - 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, i + 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if (M > 1) {
        TaskLauncher launcher(TASK_SYSTEM_LAST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, M - 1),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(runtime, ctx, M - 2),
                              READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(runtime, ctx, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier – wait for every block task to finish
    for (auto &f : futures)
        f.get_void_result();
}

/* ------------------------------------------------------------------ */
/*  Energy computation (sequential, uses inline mapping)              */
/* ------------------------------------------------------------------ */

inline double energy(const state_type &q_state, const state_type &p_state,
                     Runtime *runtime, Context ctx)
{
    RegionRequirement q_req(q_state.region, READ_ONLY, EXCLUSIVE, q_state.region);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_state.region, READ_ONLY, EXCLUSIVE, p_state.region);
    p_req.add_field(FID_VAL);

    InlineLauncher q_launcher(q_req);
    InlineLauncher p_launcher(p_req);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_launcher);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_launcher);
    q_phys.wait_until_valid();
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        q_state.region.get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double en = 0.5 * pow(abs(q_acc[lo]), LAMBDA) / LAMBDA;
    for (coord_t i = lo; i < hi; i++) {
        en += 0.5 * p_acc[i] * p_acc[i]
            + pow(q_acc[i], KAPPA) / KAPPA
            + pow(abs(q_acc[i] - q_acc[i + 1]), LAMBDA) / LAMBDA;
    }
    en += 0.5 * p_acc[hi] * p_acc[hi]
        + pow(q_acc[hi], KAPPA) / KAPPA
        + 0.5 * pow(abs(q_acc[hi]), LAMBDA) / LAMBDA;

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return en;
}

/* ------------------------------------------------------------------ */
/*  Task pre-registration (call from main before Runtime::start)      */
/* ------------------------------------------------------------------ */

inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_FIRST_BLOCK,
                                       "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_CENTER_BLOCK,
                                       "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_LAST_BLOCK,
                                       "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif
