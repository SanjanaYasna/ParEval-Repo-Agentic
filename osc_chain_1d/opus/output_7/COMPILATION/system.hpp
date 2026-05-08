// Copyright 2013 Mario Mulansky
// Translated to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <memory>
#include <cmath>

#include <legion.h>
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

typedef std::vector<double> dvec;

// Field ID for double values stored in logical regions
enum FieldIDs {
    FID_VAL = 0,
};

// Task IDs used by system tasks
enum SystemTaskIDs {
    GET_FIRST_ELEM_TASK_ID = 20,
    GET_LAST_ELEM_TASK_ID,
    FIRST_BLOCK_TASK_ID,
    CENTER_BLOCK_TASK_ID,
    LAST_BLOCK_TASK_ID,
};

// Legion state type: a partitioned logical region
struct state_type {
    LogicalRegion lr;
    LogicalPartition lp;
    size_t num_parts;   // M
    size_t block_size;  // G
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

// -------- Boundary extraction tasks --------

inline double get_first_element_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.lo];
}

inline double get_last_element_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.hi];
}

// -------- Block computation tasks --------

// First block:
//   Region[0]: q sub-region (READ_ONLY)
//   Region[1]: dpdt sub-region (WRITE_DISCARD)
//   Future[0]: q_r (first element of right neighbor block)
inline void system_first_block_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    double q_r = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    size_t N = rect.volume();
    coord_t base = rect.lo[0];

    double coupling_lr = -signed_pow(q_acc[base], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i)
    {
        dpdt_acc[base + (coord_t)i] = -signed_pow(q_acc[base + (coord_t)i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[base + (coord_t)i] - q_acc[base + (coord_t)i + 1], LAMBDA - 1);
        dpdt_acc[base + (coord_t)i] -= coupling_lr;
    }
    dpdt_acc[base + (coord_t)(N - 1)] = -signed_pow(q_acc[base + (coord_t)(N - 1)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[base + (coord_t)(N - 1)] - q_r, LAMBDA - 1);
}

// Center block:
//   Region[0]: q sub-region (READ_ONLY)
//   Region[1]: dpdt sub-region (WRITE_DISCARD)
//   Future[0]: q_l (last element of left neighbor block)
//   Future[1]: q_r (first element of right neighbor block)
inline void system_center_block_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();
    double q_r = task->futures[1].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    size_t N = rect.volume();
    coord_t base = rect.lo[0];

    double coupling_lr = -signed_pow(q_acc[base] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i)
    {
        dpdt_acc[base + (coord_t)i] = -signed_pow(q_acc[base + (coord_t)i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[base + (coord_t)i] - q_acc[base + (coord_t)i + 1], LAMBDA - 1);
        dpdt_acc[base + (coord_t)i] -= coupling_lr;
    }
    dpdt_acc[base + (coord_t)(N - 1)] = -signed_pow(q_acc[base + (coord_t)(N - 1)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[base + (coord_t)(N - 1)] - q_r, LAMBDA - 1);
}

// Last block:
//   Region[0]: q sub-region (READ_ONLY)
//   Region[1]: dpdt sub-region (WRITE_DISCARD)
//   Future[0]: q_l (last element of left neighbor block)
inline void system_last_block_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    size_t N = rect.volume();
    coord_t base = rect.lo[0];

    double coupling_lr = -signed_pow(q_acc[base] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i)
    {
        dpdt_acc[base + (coord_t)i] = -signed_pow(q_acc[base + (coord_t)i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[base + (coord_t)i] - q_acc[base + (coord_t)i + 1], LAMBDA - 1);
        dpdt_acc[base + (coord_t)i] -= coupling_lr;
    }
    dpdt_acc[base + (coord_t)(N - 1)] = -signed_pow(q_acc[base + (coord_t)(N - 1)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[base + (coord_t)(N - 1)], LAMBDA - 1);
}

// -------- osc_chain: launch Legion tasks to compute dpdt from q --------

inline void osc_chain(state_type &q, state_type &dpdt,
                      Runtime *runtime, Context ctx)
{
    const size_t M = q.num_parts;

    // Extract boundary elements from each q block as futures
    std::vector<Future> q_first(M);
    std::vector<Future> q_last(M);

    for (size_t i = 0; i < M; ++i)
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));

        {
            TaskLauncher launcher(GET_FIRST_ELEM_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            q_first[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(GET_LAST_ELEM_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            q_last[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // First block
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>(0)));
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>(0)));

        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(q_first[1]); // q_r = first element of block 1
        runtime->execute_task(ctx, launcher);
    }

    // Middle blocks
    for (size_t i = 1; i < M - 1; ++i)
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)i)));

        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(q_last[i - 1]);   // q_l = last element of left neighbor
        launcher.add_future(q_first[i + 1]);   // q_r = first element of right neighbor
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)(M - 1))));
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)(M - 1))));

        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(q_last[M - 2]); // q_l = last element of left neighbor
        runtime->execute_task(ctx, launcher);
    }
}

// osc_chain with global barrier equivalent
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                         Runtime *runtime, Context ctx)
{
    osc_chain(q, dpdt, runtime, ctx);
    // In Legion, issue an execution fence to enforce ordering
    // (equivalent to HPX's wait_all on dpdt)
    runtime->issue_execution_fence(ctx);
}

// -------- Energy computation --------

inline double energy(const dvec &q, const dvec &p)
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

inline double energy(const state_type &q_st, const state_type &p_st,
                     Runtime *runtime, Context ctx)
{
    dvec q, p;

    for (size_t i = 0; i < q_st.num_parts; ++i)
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q_st.lp, DomainPoint(Point<1>((coord_t)i)));
        LogicalRegion p_sub = runtime->get_logical_subregion_by_color(
            p_st.lp, DomainPoint(Point<1>((coord_t)i)));

        InlineLauncher q_il(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_st.lr));
        q_il.requirement.add_field(FID_VAL);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
        q_pr.wait_until_valid();

        InlineLauncher p_il(
            RegionRequirement(p_sub, READ_ONLY, EXCLUSIVE, p_st.lr));
        p_il.requirement.add_field(FID_VAL);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
        p_pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

        Rect<1> rect = runtime->get_index_space_domain(q_sub.get_index_space());
        for (PointInRectIterator<1> pir(rect); pir(); pir++)
        {
            q.push_back(q_acc[*pir]);
            p.push_back(p_acc[*pir]);
        }

        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    return energy(q, p);
}

// -------- Task registration --------

inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(GET_FIRST_ELEM_TASK_ID, "get_first_element");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, get_first_element_task>(
            registrar, "get_first_element");
    }
    {
        TaskVariantRegistrar registrar(GET_LAST_ELEM_TASK_ID, "get_last_element");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, get_last_element_task>(
            registrar, "get_last_element");
    }
    {
        TaskVariantRegistrar registrar(FIRST_BLOCK_TASK_ID, "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(CENTER_BLOCK_TASK_ID, "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(LAST_BLOCK_TASK_ID, "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
}

#endif
