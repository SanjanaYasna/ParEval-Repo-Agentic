// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>

#include <legion.h>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Field ID for double values stored in regions
enum FieldIDs {
    FID_VAL = 0,
};

// Task IDs for system-related tasks
enum SystemTaskIDs {
    SYSTEM_FIRST_BLOCK_TASK_ID = 10,
    SYSTEM_CENTER_BLOCK_TASK_ID = 11,
    SYSTEM_LAST_BLOCK_TASK_ID = 12,
};

// State type: a partitioned logical region representing M blocks of G doubles each
struct state_type {
    LogicalRegion lr;         // Parent logical region (N elements total)
    LogicalPartition lp;      // Partition into M blocks
    IndexSpace color_is;      // Color index space (M points)
    size_t num_blocks;        // M
    size_t block_size;        // G (elements per block)

    LogicalRegion subregion(Runtime *runtime, Context ctx, size_t i) const {
        return runtime->get_logical_subregion_by_color(
            lp, DomainPoint(Point<1>(static_cast<coord_t>(i))));
    }

    size_t size() const { return num_blocks; }
};

// Safe power function that handles zero base
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0)
            return 0.0;
        using std::pow;
        using std::abs;
        return pow(abs(x), y);
    }
}

// Signed power: |x|^k * sign(x)
inline double signed_pow(double x, double k) {
    using boost::math::sign;
    using std::abs;
    return checked_math::pow(x, k) * sign(x);
}

// ============================================================================
// Legion task implementations for oscillator chain force computation
// ============================================================================

// First block task: left boundary fixed at q=0, right boundary from q[1]
// regions[0]: q block 0     (READ_ONLY)
// regions[1]: q block 1     (READ_ONLY) — first element used as right boundary
// regions[2]: dpdt block 0  (WRITE_DISCARD)
inline void system_first_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_next_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    const coord_t base = q_rect.lo[0];
    const size_t N = static_cast<size_t>(q_rect.volume());

    // Right boundary: first element of next block
    double q_r = q_next_acc[q_next_rect.lo];

    // Left boundary coupling (left neighbor is 0): -signed_pow(q[0]-0, LAMBDA-1)
    double coupling_lr = -signed_pow(q_acc[base], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        coord_t idx = base + static_cast<coord_t>(i);
        double val = -signed_pow(q_acc[idx], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[idx] - q_acc[idx + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[idx] = val;
    }
    coord_t last = base + static_cast<coord_t>(N - 1);
    dpdt_acc[last] = -signed_pow(q_acc[last], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[last] - q_r, LAMBDA - 1);
}

// Center block task: boundaries from both neighboring blocks
// regions[0]: q block i     (READ_ONLY)
// regions[1]: q block i-1   (READ_ONLY) — last element used as left boundary
// regions[2]: q block i+1   (READ_ONLY) — first element used as right boundary
// regions[3]: dpdt block i  (WRITE_DISCARD)
inline void system_center_block_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_prev_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());
    Rect<1> q_next_rect = runtime->get_index_space_domain(ctx,
        regions[2].get_logical_region().get_index_space());

    const coord_t base = q_rect.lo[0];
    const size_t N = static_cast<size_t>(q_rect.volume());

    // Boundary values from neighbors
    double q_l = q_prev_acc[q_prev_rect.hi];
    double q_r = q_next_acc[q_next_rect.lo];

    double coupling_lr = -signed_pow(q_acc[base] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        coord_t idx = base + static_cast<coord_t>(i);
        double val = -signed_pow(q_acc[idx], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[idx] - q_acc[idx + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[idx] = val;
    }
    coord_t last = base + static_cast<coord_t>(N - 1);
    dpdt_acc[last] = -signed_pow(q_acc[last], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[last] - q_r, LAMBDA - 1);
}

// Last block task: right boundary fixed at q=0, left boundary from q[M-2]
// regions[0]: q block M-1     (READ_ONLY)
// regions[1]: q block M-2     (READ_ONLY) — last element used as left boundary
// regions[2]: dpdt block M-1  (WRITE_DISCARD)
inline void system_last_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        regions[0].get_logical_region().get_index_space());
    Rect<1> q_prev_rect = runtime->get_index_space_domain(ctx,
        regions[1].get_logical_region().get_index_space());

    const coord_t base = q_rect.lo[0];
    const size_t N = static_cast<size_t>(q_rect.volume());

    // Left boundary: last element of previous block
    double q_l = q_prev_acc[q_prev_rect.hi];

    double coupling_lr = -signed_pow(q_acc[base] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        coord_t idx = base + static_cast<coord_t>(i);
        double val = -signed_pow(q_acc[idx], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[idx] - q_acc[idx + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[idx] = val;
    }
    coord_t last = base + static_cast<coord_t>(N - 1);
    dpdt_acc[last] = -signed_pow(q_acc[last], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[last], LAMBDA - 1);
}

// ============================================================================
// Oscillator chain system function — launches non-blocking tasks
// (analogous to HPX dataflow graph construction in original osc_chain)
// ============================================================================

inline void osc_chain(state_type &q, state_type &dpdt,
                       Context ctx, Runtime *runtime) {
    const size_t M = q.num_blocks;

    // First block: reads q[0] and q[1], writes dpdt[0]
    {
        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, 0),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks: reads q[i-1], q[i], q[i+1], writes dpdt[i]
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i + 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block: reads q[M-2] and q[M-1], writes dpdt[M-1]
    if (M > 1) {
        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, M - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, M - 2),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// Version with global barrier (analogous to osc_chain_gb with wait_all)
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                          Context ctx, Runtime *runtime) {
    const size_t M = q.num_blocks;
    std::vector<Future> futures;

    // First block
    {
        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, 0),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, i + 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if (M > 1) {
        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, M - 1),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.subregion(runtime, ctx, M - 2),
                              READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.subregion(runtime, ctx, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier: wait for all launched tasks to complete
    for (auto &f : futures) {
        f.get_void_result();
    }
}

// ============================================================================
// Energy computation
// ============================================================================

// Energy on raw contiguous vectors (unchanged from original)
inline double energy(const std::vector<double> &q, const std::vector<double> &p) {
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
            + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

// Energy computation on Legion state_type via inline mapping
// (replaces the HPX template that called .get() on futures)
inline double energy(const state_type &q_state, const state_type &p_state,
                      Context ctx, Runtime *runtime) {
    // Inline map the full parent regions — this acts as an implicit barrier,
    // waiting for any outstanding tasks that write to these regions.
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    InlineLauncher q_launcher(q_req);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_launcher);
    q_phys.wait_until_valid();

    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);
    InlineLauncher p_launcher(p_req);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_launcher);
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        q_state.lr.get_index_space());
    const size_t N = static_cast<size_t>(rect.volume());

    // Gather into contiguous vectors and reuse the vector-based energy function
    std::vector<double> q_vec(N), p_vec(N);
    for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
        size_t j = static_cast<size_t>(i - rect.lo[0]);
        q_vec[j] = q_acc[i];
        p_vec[j] = p_acc[i];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q_vec, p_vec);
}

// ============================================================================
// Task registration (call before Runtime::start in main)
// ============================================================================

inline void register_system_tasks() {
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

#endif
