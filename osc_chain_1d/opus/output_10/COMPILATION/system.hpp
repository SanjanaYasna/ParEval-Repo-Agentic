// Copyright 2013 Mario Mulansky

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

// Field IDs shared across the codebase
enum FieldIDs {
    FID_VAL = 101,
};

// Task IDs for system tasks
enum SystemTaskIDs {
    FIRST_BLOCK_TASK_ID = 10,
    CENTER_BLOCK_TASK_ID = 11,
    LAST_BLOCK_TASK_ID = 12,
};

// Legion state type: a partitioned logical region
struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    Runtime *runtime;
    Context ctx;
    size_t M;  // number of blocks (partitions)
    size_t G;  // elements per block

    size_t size() const { return M; }

    LogicalRegion get_subregion(size_t i) const {
        return runtime->get_logical_subregion_by_color(partition, Point<1>(i));
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

// ---- Legion task: first block (left boundary) ----
void first_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    // regions[0]: q current block   (READ_ONLY)
    // regions[1]: q next block      (READ_ONLY) – need its first element
    // regions[2]: dpdt current block (WRITE_DISCARD)

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_next_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_r = q_next_acc[q_next_rect.lo];

    double coupling_lr = -signed_pow(q_acc[lo], LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        dpdt_acc[i] = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        dpdt_acc[i] -= coupling_lr;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// ---- Legion task: center block (interior) ----
void center_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    // regions[0]: q current block   (READ_ONLY)
    // regions[1]: q left block      (READ_ONLY) – need its last element
    // regions[2]: q right block     (READ_ONLY) – need its first element
    // regions[3]: dpdt current block (WRITE_DISCARD)

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_left_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());
    Rect<1> q_right_rect = runtime->get_index_space_domain(ctx,
        regions[2].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_l = q_left_acc[q_left_rect.hi];   // last element of left block
    double q_r = q_right_acc[q_right_rect.lo];  // first element of right block

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        dpdt_acc[i] = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        dpdt_acc[i] -= coupling_lr;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
}

// ---- Legion task: last block (right boundary) ----
void last_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    // regions[0]: q current block   (READ_ONLY)
    // regions[1]: q left block      (READ_ONLY) – need its last element
    // regions[2]: dpdt current block (WRITE_DISCARD)

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_left_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    coord_t lo = q_rect.lo[0];
    coord_t hi = q_rect.hi[0];

    double q_l = q_left_acc[q_left_rect.hi];  // last element of left block

    double coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; ++i)
    {
        dpdt_acc[i] = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        dpdt_acc[i] -= coupling_lr;
    }
    dpdt_acc[hi] = -signed_pow(q_acc[hi], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[hi], LAMBDA - 1);
}

// ---- Launch all block tasks (non-blocking, Legion orders by regions) ----
void osc_chain(state_type &q, state_type &dpdt)
{
    Runtime *runtime = q.runtime;
    Context ctx = q.ctx;
    const size_t M = q.M;

    assert(M >= 2);

    // First block
    {
        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(0), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(1, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(0), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(2, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(i), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(i - 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(1, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(i + 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(2, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(i), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(3, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(M - 1), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.get_subregion(M - 2), READ_ONLY, EXCLUSIVE, q.region));
        launcher.add_field(1, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.get_subregion(M - 1), WRITE_DISCARD, EXCLUSIVE, dpdt.region));
        launcher.add_field(2, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ---- Global-barrier version ----
void osc_chain_gb(state_type &q, state_type &dpdt)
{
    osc_chain(q, dpdt);
    // Execution fence: all tasks launched before the fence complete
    // before any task launched after the fence may begin.
    Runtime *runtime = q.runtime;
    Context ctx = q.ctx;
    runtime->issue_execution_fence(ctx);
}

// ---- Energy on flat vectors (unchanged physics) ----
double energy(const std::vector<double> &q, const std::vector<double> &p)
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

// ---- Energy for state_type: inline-map both regions, gather, compute ----
double energy(const state_type &q_state, const state_type &p_state)
{
    Runtime *runtime = q_state.runtime;
    Context ctx = q_state.ctx;

    // Inline-map q
    RegionRequirement q_req(q_state.region, READ_ONLY, EXCLUSIVE, q_state.region);
    q_req.add_field(FID_VAL);
    InlineLauncher q_il(q_req);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_il);
    q_phys.wait_until_valid();

    // Inline-map p
    RegionRequirement p_req(p_state.region, READ_ONLY, EXCLUSIVE, p_state.region);
    p_req.add_field(FID_VAL);
    InlineLauncher p_il(p_req);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_il);
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        q_state.region.get_index_space());
    coord_t lo = rect.lo[0];
    size_t N = rect.volume();

    // Gather into contiguous vectors
    std::vector<double> q(N), p(N);
    for (size_t i = 0; i < N; ++i)
    {
        q[i] = q_acc[lo + (coord_t)i];
        p[i] = p_acc[lo + (coord_t)i];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q, p);
}

// ---- Pre-register the three block tasks with the Legion runtime ----
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(FIRST_BLOCK_TASK_ID, "first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<first_block_task>(
            registrar, "first_block");
    }
    {
        TaskVariantRegistrar registrar(CENTER_BLOCK_TASK_ID, "center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<center_block_task>(
            registrar, "center_block");
    }
    {
        TaskVariantRegistrar registrar(LAST_BLOCK_TASK_ID, "last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<last_block_task>(
            registrar, "last_block");
    }
}

#endif
