// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>

#include <boost/math/special_functions/sign.hpp>

#include "legion.h"

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ---- Field and Task IDs ----

enum FieldIDs {
    FID_VAL = 0,
};

enum SystemTaskIDs {
    SYSTEM_FIRST_BLOCK_TASK_ID = 20,
    SYSTEM_CENTER_BLOCK_TASK_ID = 21,
    SYSTEM_LAST_BLOCK_TASK_ID = 22,
};

// ---- State type for Legion ----

typedef std::vector<double> dvec;

struct state_type {
    LogicalRegion lr;
    LogicalPartition lp;
    IndexSpace color_is;
    size_t num_blocks;   // M
    size_t block_size;   // G
    size_t total_size;   // N

    size_t size() const { return num_blocks; }
};

// ---- Math helpers ----

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

// ---- Task implementations ----

// First block: regions[0]=q[0] (RO), regions[1]=q[1] (RO), regions[2]=dpdt[0] (WD)
inline void system_first_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_next = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_r = q_next[rect_next.lo];

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

// Center block: regions[0]=q[i] (RO), regions[1]=q[i-1] (RO),
//               regions[2]=q[i+1] (RO), regions[3]=dpdt[i] (WD)
inline void system_center_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_prev = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> rect_next = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l = q_prev[rect_prev.hi];
    double q_r = q_next[rect_next.lo];

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

// Last block: regions[0]=q[M-1] (RO), regions[1]=q[M-2] (RO),
//             regions[2]=dpdt[M-1] (WD)
inline void system_last_block_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_prev = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l = q_prev[rect_prev.hi];

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

// ---- System function: launches block tasks with region requirements ----

inline void osc_chain(state_type &q, state_type &dpdt,
                      Context ctx, Runtime *runtime)
{
    const size_t M = q.num_blocks;

    // First block: reads q[0], q[1]; writes dpdt[0]
    {
        LogicalRegion q0 = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)0);
        LogicalRegion q1 = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)1);
        LogicalRegion d0 = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)0);

        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q0, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q1, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(d0, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks: reads q[i-1], q[i], q[i+1]; writes dpdt[i]
    for (size_t i = 1; i < M - 1; ++i)
    {
        LogicalRegion qi      = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)i);
        LogicalRegion qi_prev = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(i - 1));
        LogicalRegion qi_next = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(i + 1));
        LogicalRegion di      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)i);

        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(qi, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qi_prev, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qi_next, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(di, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[3].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block: reads q[M-2], q[M-1]; writes dpdt[M-1]
    if (M >= 2)
    {
        LogicalRegion qm      = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(M - 1));
        LogicalRegion qm_prev = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(M - 2));
        LogicalRegion dm      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)(M - 1));

        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(qm, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qm_prev, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dm, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// Version with global barrier (wait for all block tasks to finish)
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                         Context ctx, Runtime *runtime)
{
    const size_t M = q.num_blocks;
    std::vector<Future> futures;

    // First block
    {
        LogicalRegion q0 = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)0);
        LogicalRegion q1 = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)1);
        LogicalRegion d0 = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)0);

        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q0, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q1, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(d0, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        LogicalRegion qi      = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)i);
        LogicalRegion qi_prev = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(i - 1));
        LogicalRegion qi_next = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(i + 1));
        LogicalRegion di      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)i);

        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(qi, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qi_prev, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qi_next, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(di, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[3].add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if (M >= 2)
    {
        LogicalRegion qm      = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(M - 1));
        LogicalRegion qm_prev = runtime->get_logical_subregion_by_color(ctx, q.lp, (Color)(M - 2));
        LogicalRegion dm      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp, (Color)(M - 1));

        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(qm, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(qm_prev, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dm, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier: wait for all tasks
    for (auto &f : futures)
        f.get_void_result();
}

// Functor wrapper so osc_chain can be passed to integrators that
// expect signature  void(state_type&, state_type&)
struct osc_chain_system
{
    Context ctx;
    Runtime *runtime;

    osc_chain_system(Context c, Runtime *r) : ctx(c), runtime(r) {}

    void operator()(state_type &q, state_type &dpdt) const
    {
        osc_chain(q, dpdt, ctx, runtime);
    }
};

// ---- Energy computation ----

inline double energy(const dvec &q, const dvec &p)
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

// Energy from Legion state_type: inline-maps regions and gathers data
inline double energy(const state_type &q_state, const state_type &p_state,
                     Context ctx, Runtime *runtime)
{
    // Inline-map the full logical regions
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    InlineLauncher q_il(q_req);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_il);
    q_phys.wait_until_valid();

    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);
    InlineLauncher p_il(p_req);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_il);
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    const size_t N = q_state.total_size;

    // Gather into local vectors and delegate to the dvec version
    dvec q_vec(N), p_vec(N);
    for (size_t i = 0; i < N; ++i)
    {
        q_vec[i] = q_acc[Point<1>(i)];
        p_vec[i] = p_acc[Point<1>(i)];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q_vec, p_vec);
}

// ---- Task registration ----

inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(SYSTEM_FIRST_BLOCK_TASK_ID, "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_CENTER_BLOCK_TASK_ID, "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_LAST_BLOCK_TASK_ID, "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif
