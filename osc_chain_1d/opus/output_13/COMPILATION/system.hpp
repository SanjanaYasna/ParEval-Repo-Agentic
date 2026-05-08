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

// Field and Task IDs shared across translation units
#ifndef FIELD_IDS_DEFINED
#define FIELD_IDS_DEFINED
enum FieldIDs {
    FID_VAL = 101,
};
#endif

enum SystemTaskIDs {
    SYSTEM_BLOCK_TASK_ID = 201,
    ENERGY_TASK_ID = 202,
};

// Legion state type: a 1D region of doubles, equally partitioned into M blocks
struct state_type {
    LogicalRegion region;
    LogicalPartition partition;
    IndexSpace color_space;
    size_t N; // total number of elements
    size_t G; // elements per block
    size_t M; // number of blocks (N/G)
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

// Task argument for system block computation
struct SystemBlockArgs {
    coord_t block_index;
    coord_t num_blocks;
    bool has_left;
    bool has_right;
};

// Task: compute dpdt for one block of the oscillator chain
// Region 0: own q sub-region (READ_ONLY)
// Region 1: own dpdt sub-region (WRITE_DISCARD)
// Region 2: left neighbor q sub-region (READ_ONLY), if has_left
// Region 2 or 3: right neighbor q sub-region (READ_ONLY), if has_right
inline void system_block_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    const SystemBlockArgs &args = *(const SystemBlockArgs *)task->args;
    const coord_t bi = args.block_index;
    const coord_t M  = args.num_blocks;

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Extract boundary values from neighbor blocks
    double q_l = 0.0;
    double q_r = 0.0;
    int ridx = 2;

    if (args.has_left) {
        const FieldAccessor<READ_ONLY, double, 1> ql_acc(regions[ridx], FID_VAL);
        Rect<1> lr = runtime->get_index_space_domain(
            regions[ridx].get_logical_region().get_index_space());
        q_l = ql_acc[lr.hi[0]]; // last element of left neighbor
        ridx++;
    }
    if (args.has_right) {
        const FieldAccessor<READ_ONLY, double, 1> qr_acc(regions[ridx], FID_VAL);
        Rect<1> rr = runtime->get_index_space_domain(
            regions[ridx].get_logical_region().get_index_space());
        q_r = qr_acc[rr.lo[0]]; // first element of right neighbor
    }

    // Compute initial left coupling
    // First block: left boundary fixed at 0, coupling = -signed_pow(q[0], LAMBDA-1)
    // Other blocks: coupling = -signed_pow(q[0] - q_l, LAMBDA-1)
    double coupling_lr;
    if (bi == 0)
        coupling_lr = -signed_pow(q_acc[lo], LAMBDA - 1);
    else
        coupling_lr = -signed_pow(q_acc[lo] - q_l, LAMBDA - 1);

    // Interior loop
    for (coord_t j = lo; j < hi; j++) {
        double val = -signed_pow(q_acc[j], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[j] - q_acc[j + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[j] = val;
    }

    // Last element of block
    // Last block: right boundary fixed at 0, coupling = signed_pow(q[N-1], LAMBDA-1)
    // Other blocks: coupling = signed_pow(q[N-1] - q_r, LAMBDA-1)
    double last_val = -signed_pow(q_acc[hi], KAPPA - 1) + coupling_lr;
    if (bi == M - 1)
        last_val -= signed_pow(q_acc[hi], LAMBDA - 1);
    else
        last_val -= signed_pow(q_acc[hi] - q_r, LAMBDA - 1);
    dpdt_acc[hi] = last_val;
}

// Helper to launch a single system block task
static inline Future launch_system_block(
    size_t i, size_t M,
    state_type &q, state_type &dpdt,
    Runtime *runtime, Context ctx)
{
    SystemBlockArgs args;
    args.block_index = (coord_t)i;
    args.num_blocks  = (coord_t)M;
    args.has_left    = (i > 0);
    args.has_right   = (i < M - 1);

    TaskLauncher launcher(SYSTEM_BLOCK_TASK_ID,
                          TaskArgument(&args, sizeof(args)));

    // Region 0: own q sub-region (READ_ONLY)
    LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
        q.partition, DomainPoint(Point<1>((coord_t)i)));
    launcher.add_region_requirement(
        RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.region));
    launcher.region_requirements.back().add_field(FID_VAL);

    // Region 1: own dpdt sub-region (WRITE_DISCARD)
    LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
        dpdt.partition, DomainPoint(Point<1>((coord_t)i)));
    launcher.add_region_requirement(
        RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.region));
    launcher.region_requirements.back().add_field(FID_VAL);

    // Region 2: left neighbor q sub-region (READ_ONLY)
    if (i > 0) {
        LogicalRegion q_left = runtime->get_logical_subregion_by_color(
            q.partition, DomainPoint(Point<1>((coord_t)(i - 1))));
        launcher.add_region_requirement(
            RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
    }

    // Region 3 (or 2): right neighbor q sub-region (READ_ONLY)
    if (i < M - 1) {
        LogicalRegion q_right = runtime->get_logical_subregion_by_color(
            q.partition, DomainPoint(Point<1>((coord_t)(i + 1))));
        launcher.add_region_requirement(
            RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, q.region));
        launcher.region_requirements.back().add_field(FID_VAL);
    }

    return runtime->execute_task(ctx, launcher);
}

// Launch the oscillator chain system computation (non-blocking).
// Legion runtime tracks dependencies through region requirements.
inline void osc_chain(state_type &q, state_type &dpdt,
                      Runtime *runtime, Context ctx)
{
    const size_t M = q.M;
    for (size_t i = 0; i < M; i++) {
        launch_system_block(i, M, q, dpdt, runtime, ctx);
    }
}

// Launch the oscillator chain system computation with a global barrier.
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                         Runtime *runtime, Context ctx)
{
    const size_t M = q.M;
    std::vector<Future> futures(M);

    for (size_t i = 0; i < M; i++) {
        futures[i] = launch_system_block(i, M, q, dpdt, runtime, ctx);
    }

    // Global barrier: wait for all block tasks to complete
    for (size_t i = 0; i < M; i++) {
        futures[i].get_void_result();
    }
}

// Task: compute total energy over the entire oscillator chain
// Region 0: entire q region (READ_ONLY)
// Region 1: entire p region (READ_ONLY)
inline double energy_task_impl(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double e = 0.5 * pow(abs(q[lo]), LAMBDA) / LAMBDA;
    for (coord_t i = lo; i < hi; i++) {
        e += 0.5 * p[i] * p[i]
           + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[hi] * p[hi]
       + pow(q[hi], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[hi]), LAMBDA) / LAMBDA;

    return e;
}

// Compute and return total energy (blocking call, launches a single task)
inline double energy(const state_type &q, const state_type &p,
                     Runtime *runtime, Context ctx)
{
    TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));

    launcher.add_region_requirement(
        RegionRequirement(q.region, READ_ONLY, EXCLUSIVE, q.region));
    launcher.region_requirements.back().add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(p.region, READ_ONLY, EXCLUSIVE, p.region));
    launcher.region_requirements.back().add_field(FID_VAL);

    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

// Register all system-related tasks with the Legion runtime.
// Must be called before Runtime::start().
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_block_task>(
            registrar, "system_block");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, energy_task_impl>(
            registrar, "energy");
    }
}

#endif
