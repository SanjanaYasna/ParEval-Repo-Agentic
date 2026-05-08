// Copyright 2013 Mario Mulansky
// Translated to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <legion.h>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum SystemTaskIDs {
    TASK_SYSTEM_FIRST_BLOCK = 10,
    TASK_SYSTEM_CENTER_BLOCK = 11,
    TASK_SYSTEM_LAST_BLOCK = 12,
    TASK_COMPUTE_ENERGY = 13,
};

enum FieldIDs {
    FID_VAL = 101,
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
    double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow( x , k ) * s;
}

// ========================================================================
// First block task
// Regions: [0] q own (READ), [1] q right neighbor (READ), [2] dpdt own (WRITE)
// ========================================================================
void system_first_block_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_r_val = q_right[right_rect.lo]; // first element of right neighbor

    double coupling_lr = -signed_pow( q[Point<1>(lo)] , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        dpdt[Point<1>(i)] = val - coupling_lr;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] - q_r_val , LAMBDA-1 );
}

// ========================================================================
// Center block task
// Regions: [0] q own (READ), [1] q left neighbor (READ),
//          [2] q right neighbor (READ), [3] dpdt own (WRITE)
// ========================================================================
void system_center_block_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> right_rect = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l_val = q_left[left_rect.hi];   // last element of left neighbor
    double q_r_val = q_right[right_rect.lo];  // first element of right neighbor

    double coupling_lr = -signed_pow( q[Point<1>(lo)] - q_l_val , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        dpdt[Point<1>(i)] = val - coupling_lr;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] - q_r_val , LAMBDA-1 );
}

// ========================================================================
// Last block task
// Regions: [0] q own (READ), [1] q left neighbor (READ), [2] dpdt own (WRITE)
// ========================================================================
void system_last_block_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    double q_l_val = q_left[left_rect.hi]; // last element of left neighbor

    double coupling_lr = -signed_pow( q[Point<1>(lo)] - q_l_val , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        dpdt[Point<1>(i)] = val - coupling_lr;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] , LAMBDA-1 );
}

// ========================================================================
// Energy computation task
// Regions: [0] q (full region, READ), [1] p (full region, READ)
// Returns total energy as double
// ========================================================================
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double e = 0.5 * pow( abs((double)q[Point<1>(lo)]) , LAMBDA ) / LAMBDA;
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double qi  = q[Point<1>(i)];
        double qi1 = q[Point<1>(i+1)];
        double pi  = p[Point<1>(i)];
        e += 0.5*pi*pi + pow( qi , KAPPA ) / KAPPA
            + pow( abs(qi - qi1) , LAMBDA ) / LAMBDA;
    }
    double qhi = q[Point<1>(hi)];
    double phi = p[Point<1>(hi)];
    e += 0.5*phi*phi + pow( qhi , KAPPA ) / KAPPA
        + 0.5 * pow( abs(qhi) , LAMBDA ) / LAMBDA;

    return e;
}

// ========================================================================
// Launch the oscillator chain system: computes dpdt = f(q) for all blocks
// Legion runtime handles task ordering via region requirements.
// ========================================================================
void osc_chain(Context ctx, Runtime *runtime,
               LogicalRegion q_lr, LogicalPartition q_lp,
               LogicalRegion dpdt_lr, LogicalPartition dpdt_lp,
               size_t M)
{
    assert(M >= 2);

    for( size_t i = 0 ; i < M ; ++i )
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            ctx, q_lp, DomainPoint(Point<1>(i)));
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
            ctx, dpdt_lp, DomainPoint(Point<1>(i)));

        if( i == 0 )
        {
            // First block: reads own q + right neighbor q, writes own dpdt
            LogicalRegion q_right = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(1)));

            TaskLauncher launcher(TASK_SYSTEM_FIRST_BLOCK, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        else if( i == M - 1 )
        {
            // Last block: reads own q + left neighbor q, writes own dpdt
            LogicalRegion q_left = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(i - 1)));

            TaskLauncher launcher(TASK_SYSTEM_LAST_BLOCK, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        else
        {
            // Center block: reads own q + left + right neighbor q, writes own dpdt
            LogicalRegion q_left = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(i - 1)));
            LogicalRegion q_right = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(i + 1)));

            TaskLauncher launcher(TASK_SYSTEM_CENTER_BLOCK, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }
}

// ========================================================================
// Compute and return total energy (launches energy task, blocks on result)
// ========================================================================
double energy(Context ctx, Runtime *runtime,
              LogicalRegion q_lr, LogicalRegion p_lr)
{
    TaskLauncher launcher(TASK_COMPUTE_ENERGY, TaskArgument(NULL, 0));
    launcher.add_region_requirement(
        RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    launcher.region_requirements.back().add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    launcher.region_requirements.back().add_field(FID_VAL);
    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

// ========================================================================
// Register all system-related tasks (call before Runtime::start)
// ========================================================================
void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_FIRST_BLOCK, "system_first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_first_block_task>(
            registrar, "system_first_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_CENTER_BLOCK, "system_center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_center_block_task>(
            registrar, "system_center_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_SYSTEM_LAST_BLOCK, "system_last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<system_last_block_task>(
            registrar, "system_last_block");
    }
    {
        TaskVariantRegistrar registrar(TASK_COMPUTE_ENERGY, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, energy_task>(
            registrar, "energy");
    }
}

#endif
