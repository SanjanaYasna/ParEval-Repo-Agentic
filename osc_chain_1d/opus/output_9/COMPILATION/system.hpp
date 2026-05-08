// Copyright 2013 Mario Mulansky
// Translated to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>

#include <boost/math/special_functions/sign.hpp>

#include "legion.h"
#include "shared_resize.hpp"

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// System-specific task IDs
enum SystemTaskIDs {
    SYSTEM_FIRST_BLOCK_TASK_ID = 20,
    SYSTEM_CENTER_BLOCK_TASK_ID = 21,
    SYSTEM_LAST_BLOCK_TASK_ID = 22,
    EXTRACT_FIRST_ELEM_TASK_ID = 23,
    EXTRACT_LAST_ELEM_TASK_ID = 24,
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

double signed_pow( double x , double k )
{
    using boost::math::sign;
    using std::abs;
    return checked_math::pow( x , k ) * sign(x);
}

typedef std::vector< double > dvec;

// ---- Legion Task Implementations ----

// Extract the first element of a sub-region (replaces the HPX lambda extracting (*v)[0])
double extract_first_elem_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    return acc[rect.lo];
}

// Extract the last element of a sub-region (replaces the HPX lambda extracting (*v)[v->size()-1])
double extract_last_elem_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    return acc[rect.hi];
}

// First block computation task
// Region 0: q sub-region (READ_ONLY), Region 1: dpdt sub-region (WRITE_DISCARD)
// Future 0: q_r (first element of next block)
void system_first_block_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    double q_r = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N), dpdt(N);
    for( size_t i = 0 ; i < N ; i++ )
        q[i] = q_acc[Point<1>(lo + (coord_t)i)];

    double coupling_lr = -signed_pow( q[0] , LAMBDA-1 );
    for( size_t i = 0 ; i < N-1 ; i++ )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[N-1] = -signed_pow( q[N-1] , KAPPA-1 )
        + coupling_lr - signed_pow( q[N-1] - q_r , LAMBDA-1 );

    for( size_t i = 0 ; i < N ; i++ )
        dpdt_acc[Point<1>(lo + (coord_t)i)] = dpdt[i];
}

// Center block computation task
// Region 0: q sub-region (READ_ONLY), Region 1: dpdt sub-region (WRITE_DISCARD)
// Future 0: q_l (last element of previous block)
// Future 1: q_r (first element of next block)
void system_center_block_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();
    double q_r = task->futures[1].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N), dpdt(N);
    for( size_t i = 0 ; i < N ; i++ )
        q[i] = q_acc[Point<1>(lo + (coord_t)i)];

    double coupling_lr = -signed_pow( q[0] - q_l , LAMBDA-1 );
    for( size_t i = 0 ; i < N-1 ; i++ )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[N-1] = -signed_pow( q[N-1] , KAPPA-1 )
        + coupling_lr - signed_pow( q[N-1] - q_r , LAMBDA-1 );

    for( size_t i = 0 ; i < N ; i++ )
        dpdt_acc[Point<1>(lo + (coord_t)i)] = dpdt[i];
}

// Last block computation task
// Region 0: q sub-region (READ_ONLY), Region 1: dpdt sub-region (WRITE_DISCARD)
// Future 0: q_l (last element of previous block)
void system_last_block_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    double q_l = task->futures[0].get_result<double>();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N), dpdt(N);
    for( size_t i = 0 ; i < N ; i++ )
        q[i] = q_acc[Point<1>(lo + (coord_t)i)];

    double coupling_lr = -signed_pow( q[0] - q_l , LAMBDA-1 );
    for( size_t i = 0 ; i < N-1 ; i++ )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[N-1] = -signed_pow( q[N-1] , KAPPA-1 )
        + coupling_lr - signed_pow( q[N-1] , LAMBDA-1 );

    for( size_t i = 0 ; i < N ; i++ )
        dpdt_acc[Point<1>(lo + (coord_t)i)] = dpdt[i];
}

// ---- System Functions ----

// Helper: launch all osc_chain tasks and return compute futures
std::vector<Future> launch_osc_chain(state_type &q , state_type &dpdt)
{
    Runtime *runtime = q.runtime;
    Context ctx = q.ctx;
    const size_t M = q.num_blocks;
    std::vector<Future> compute_futures;

    // Extract boundary elements as futures from each block
    std::vector<Future> first_elem(M), last_elem(M);
    for( size_t i = 0 ; i < M ; i++ )
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));
        {
            TaskLauncher launcher(EXTRACT_FIRST_ELEM_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            first_elem[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(EXTRACT_LAST_ELEM_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            last_elem[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // First block
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>(0)));
        LogicalRegion d_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>(0)));

        TaskLauncher launcher(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_future(first_elem[1]); // q_r
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(d_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        compute_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Middle blocks
    for( size_t i = 1 ; i < M-1 ; i++ )
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));
        LogicalRegion d_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)i)));

        TaskLauncher launcher(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_future(last_elem[i-1]);   // q_l
        launcher.add_future(first_elem[i+1]);  // q_r
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(d_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        compute_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if( M > 1 )
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)(M-1))));
        LogicalRegion d_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)(M-1))));

        TaskLauncher launcher(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_future(last_elem[M-2]); // q_l
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(d_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        compute_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    return compute_futures;
}

// Non-blocking system function (tasks launched, runtime handles ordering)
void osc_chain( state_type &q , state_type &dpdt )
{
    launch_osc_chain(q, dpdt);
}

// System function with global barrier (waits for all tasks to complete)
void osc_chain_gb( state_type &q , state_type &dpdt )
{
    std::vector<Future> futures = launch_osc_chain(q, dpdt);
    for( auto &f : futures )
        f.get_void_result();
}

// ---- Energy Functions ----

// Compute energy from raw vectors (unchanged from original)
double energy( const dvec &q , const dvec &p )
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double en = 0.5*pow( abs(q[0]) , LAMBDA) / LAMBDA;
    for( size_t i=0 ; i<N-1 ; ++i )
    {
        en += 0.5*p[i]*p[i] + pow( q[i] , KAPPA ) / KAPPA
            + pow( abs(q[i]-q[i+1]) , LAMBDA ) / LAMBDA;
    }
    en += 0.5*p[N-1]*p[N-1] + pow( q[N-1] , KAPPA ) / KAPPA
        + 0.5*pow( abs(q[N-1]) , LAMBDA) / LAMBDA;
    return en;
}

// Compute energy from Legion state_type (replaces the HPX future-based template)
double energy( const state_type &q_state , const state_type &p_state )
{
    Runtime *runtime = q_state.runtime;
    Context ctx = q_state.ctx;

    // Inline-map the full regions to read data synchronously
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);

    PhysicalRegion q_phys = runtime->map_region(ctx, q_req);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_req);
    q_phys.wait_until_valid();
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    Domain dom = runtime->get_index_space_domain(q_state.lr.get_index_space());
    Rect<1> rect(dom);
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    dvec q(N), p(N);
    for( size_t i = 0 ; i < N ; i++ )
    {
        q[i] = q_acc[Point<1>(lo + (coord_t)i)];
        p[i] = p_acc[Point<1>(lo + (coord_t)i)];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy( q , p );
}

// ---- Task Registration ----

void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(EXTRACT_FIRST_ELEM_TASK_ID, "extract_first_elem");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, extract_first_elem_task>(
            registrar, "extract_first_elem");
    }
    {
        TaskVariantRegistrar registrar(EXTRACT_LAST_ELEM_TASK_ID, "extract_last_elem");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, extract_last_elem_task>(
            registrar, "extract_last_elem");
    }
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
