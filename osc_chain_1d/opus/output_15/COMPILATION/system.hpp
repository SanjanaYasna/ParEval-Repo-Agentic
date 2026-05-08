// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include "legion.h"
#include <vector>
#include <memory>
#include <cmath>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Task and field IDs shared across the codebase.
// Guarded to allow consistent definition in other translated headers.
#ifndef TASK_IDS_DEFINED
#define TASK_IDS_DEFINED
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    FIRST_BLOCK_TASK_ID,
    CENTER_BLOCK_TASK_ID,
    LAST_BLOCK_TASK_ID,
    FOR_EACH3_TASK_ID,
};
#endif

#ifndef FIELD_IDS_DEFINED
#define FIELD_IDS_DEFINED
enum FieldIDs {
    FID_VAL = 0,
};
#endif

// Legion state type: a partitioned logical region with runtime context
#ifndef STATE_TYPE_DEFINED
#define STATE_TYPE_DEFINED
struct state_type {
    LogicalRegion parent_lr;
    LogicalPartition lp;
    std::vector<LogicalRegion> subregions;
    Runtime *runtime;
    Context ctx;

    size_t size() const { return subregions.size(); }
};
#endif

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

// =====================================================================
// Legion task: compute forces for the first block
//   Region 0: q[0]    (READ_ONLY)
//   Region 1: q[1]    (READ_ONLY)  — first element used as right coupling
//   Region 2: dpdt[0] (WRITE_DISCARD)
// =====================================================================
void first_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_r = q_right[right_rect.lo];

    double coupling_lr = -signed_pow(q[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi] - q_r, LAMBDA - 1);
}

// =====================================================================
// Legion task: compute forces for a center block
//   Region 0: q[i]    (READ_ONLY)
//   Region 1: q[i-1]  (READ_ONLY)  — last element used as left coupling
//   Region 2: q[i+1]  (READ_ONLY)  — first element used as right coupling
//   Region 3: dpdt[i] (WRITE_DISCARD)
// =====================================================================
void center_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        regions[2].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l = q_left[left_rect.hi];
    double q_r = q_right[right_rect.lo];

    double coupling_lr = -signed_pow(q[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi] - q_r, LAMBDA - 1);
}

// =====================================================================
// Legion task: compute forces for the last block
//   Region 0: q[M-1]    (READ_ONLY)
//   Region 1: q[M-2]    (READ_ONLY)  — last element used as left coupling
//   Region 2: dpdt[M-1] (WRITE_DISCARD)
// =====================================================================
void last_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l = q_left[left_rect.hi];

    double coupling_lr = -signed_pow(q[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow(q[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q[hi], LAMBDA - 1);
}

// =====================================================================
// osc_chain: launch system tasks for each block (non-blocking)
// Equivalent of HPX dataflow graph construction.
// =====================================================================
void osc_chain(state_type &q, state_type &dpdt)
{
    Runtime *runtime = q.runtime;
    Context ctx = q.ctx;
    const size_t M = q.size();

    // First block
    {
        RegionRequirement rr_q0(q.subregions[0], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_q0.add_field(FID_VAL);
        RegionRequirement rr_q1(q.subregions[1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_q1.add_field(FID_VAL);
        RegionRequirement rr_dp0(dpdt.subregions[0], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dp0.add_field(FID_VAL);

        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_q0);
        launcher.add_region_requirement(rr_q1);
        launcher.add_region_requirement(rr_dp0);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        RegionRequirement rr_qi(q.subregions[i], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qi.add_field(FID_VAL);
        RegionRequirement rr_ql(q.subregions[i - 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_ql.add_field(FID_VAL);
        RegionRequirement rr_qr(q.subregions[i + 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qr.add_field(FID_VAL);
        RegionRequirement rr_dpi(dpdt.subregions[i], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dpi.add_field(FID_VAL);

        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_qi);
        launcher.add_region_requirement(rr_ql);
        launcher.add_region_requirement(rr_qr);
        launcher.add_region_requirement(rr_dpi);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    if (M > 1)
    {
        RegionRequirement rr_qN(q.subregions[M - 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qN.add_field(FID_VAL);
        RegionRequirement rr_ql(q.subregions[M - 2], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_ql.add_field(FID_VAL);
        RegionRequirement rr_dpN(dpdt.subregions[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dpN.add_field(FID_VAL);

        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_qN);
        launcher.add_region_requirement(rr_ql);
        launcher.add_region_requirement(rr_dpN);
        runtime->execute_task(ctx, launcher);
    }
}

// =====================================================================
// osc_chain_gb: same as osc_chain but with a global barrier
// (waits for all launched tasks to complete before returning)
// =====================================================================
void osc_chain_gb(state_type &q, state_type &dpdt)
{
    Runtime *runtime = q.runtime;
    Context ctx = q.ctx;
    const size_t M = q.size();
    std::vector<Future> futures;

    // First block
    {
        RegionRequirement rr_q0(q.subregions[0], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_q0.add_field(FID_VAL);
        RegionRequirement rr_q1(q.subregions[1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_q1.add_field(FID_VAL);
        RegionRequirement rr_dp0(dpdt.subregions[0], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dp0.add_field(FID_VAL);

        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_q0);
        launcher.add_region_requirement(rr_q1);
        launcher.add_region_requirement(rr_dp0);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        RegionRequirement rr_qi(q.subregions[i], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qi.add_field(FID_VAL);
        RegionRequirement rr_ql(q.subregions[i - 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_ql.add_field(FID_VAL);
        RegionRequirement rr_qr(q.subregions[i + 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qr.add_field(FID_VAL);
        RegionRequirement rr_dpi(dpdt.subregions[i], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dpi.add_field(FID_VAL);

        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_qi);
        launcher.add_region_requirement(rr_ql);
        launcher.add_region_requirement(rr_qr);
        launcher.add_region_requirement(rr_dpi);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if (M > 1)
    {
        RegionRequirement rr_qN(q.subregions[M - 1], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_qN.add_field(FID_VAL);
        RegionRequirement rr_ql(q.subregions[M - 2], READ_ONLY, EXCLUSIVE, q.parent_lr);
        rr_ql.add_field(FID_VAL);
        RegionRequirement rr_dpN(dpdt.subregions[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.parent_lr);
        rr_dpN.add_field(FID_VAL);

        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(rr_qN);
        launcher.add_region_requirement(rr_ql);
        launcher.add_region_requirement(rr_dpN);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier: wait for all tasks to finish
    for (auto &f : futures)
        f.get_void_result();
}

// =====================================================================
// Energy computation on flat vectors (unchanged from HPX version)
// =====================================================================
inline double energy(const std::vector<double> &q, const std::vector<double> &p)
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i)
    {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
            + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

// =====================================================================
// Energy computation over Legion state_type.
// Inline-maps the parent regions, reads all values, and computes energy.
// The runtime ensures all pending sub-task writes are visible.
// =====================================================================
inline double energy(const state_type &q_state, const state_type &p_state)
{
    Runtime *runtime = q_state.runtime;
    Context ctx = q_state.ctx;

    // Inline map q parent region
    RegionRequirement q_req(q_state.parent_lr, READ_ONLY, EXCLUSIVE, q_state.parent_lr);
    q_req.add_field(FID_VAL);
    InlineLauncher q_il(q_req);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
    q_pr.wait_until_valid();

    // Inline map p parent region
    RegionRequirement p_req(p_state.parent_lr, READ_ONLY, EXCLUSIVE, p_state.parent_lr);
    p_req.add_field(FID_VAL);
    InlineLauncher p_il(p_req);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        q_state.parent_lr.get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    // Assemble into flat vectors and delegate to the vector-based overload
    std::vector<double> q_vec(N), p_vec(N);
    for (size_t i = 0; i < N; ++i)
    {
        q_vec[i] = q_acc[lo + (coord_t)i];
        p_vec[i] = p_acc[lo + (coord_t)i];
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return energy(q_vec, p_vec);
}

// =====================================================================
// Register all system tasks with the Legion runtime.
// Must be called before Runtime::start().
// =====================================================================
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(FIRST_BLOCK_TASK_ID, "first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<first_block_task>(registrar, "first_block");
    }
    {
        TaskVariantRegistrar registrar(CENTER_BLOCK_TASK_ID, "center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<center_block_task>(registrar, "center_block");
    }
    {
        TaskVariantRegistrar registrar(LAST_BLOCK_TASK_ID, "last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<last_block_task>(registrar, "last_block");
    }
}

#endif
