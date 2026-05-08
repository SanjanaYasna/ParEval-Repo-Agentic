// Copyright 2013 Mario Mulansky
// Translated to the Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>

#include <boost/math/special_functions/sign.hpp>

#include "legion.h"
#include "shared_resize.hpp" // for state_type, FID_VAL

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum SystemTaskIDs {
    FIRST_BLOCK_TASK_ID = 20,
    CENTER_BLOCK_TASK_ID = 21,
    LAST_BLOCK_TASK_ID = 22,
};

namespace checked_math {
    inline double pow( double x , double y )
    {
        if( x == 0.0 )
            return 0.0;
        using std::pow;
        using std::abs;
        return pow( abs(x) , y );
    }
}

inline double signed_pow( double x , double k )
{
    using boost::math::sign;
    using std::abs;
    return checked_math::pow( x , k ) * sign(x);
}

// ---- Task: first block ----
// Regions[0]: q block 0      (READ_ONLY)
// Regions[1]: q block 1      (READ_ONLY)  -- to get first element as right boundary
// Regions[2]: dpdt block 0   (WRITE_DISCARD)
inline void first_block_task( const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime )
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> next_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    coord_t next_lo = next_rect.lo[0];

    double q_r = q_next[next_lo];

    double coupling_lr = -signed_pow( q[lo] , LAMBDA - 1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[i] , KAPPA - 1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i + 1] , LAMBDA - 1 );
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA - 1 )
        + coupling_lr - signed_pow( q[hi] - q_r , LAMBDA - 1 );
}

// ---- Task: center block ----
// Regions[0]: q block i      (READ_ONLY)
// Regions[1]: q block i-1    (READ_ONLY)  -- to get last element as left boundary
// Regions[2]: q block i+1    (READ_ONLY)  -- to get first element as right boundary
// Regions[3]: dpdt block i   (WRITE_DISCARD)
inline void center_block_task( const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime )
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> prev_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());
    Rect<1> next_rect = runtime->get_index_space_domain(
        regions[2].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    coord_t prev_hi = prev_rect.hi[0];
    coord_t next_lo = next_rect.lo[0];

    double q_l = q_prev[prev_hi];
    double q_r = q_next[next_lo];

    double coupling_lr = -signed_pow( q[lo] - q_l , LAMBDA - 1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[i] , KAPPA - 1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i + 1] , LAMBDA - 1 );
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA - 1 )
        + coupling_lr - signed_pow( q[hi] - q_r , LAMBDA - 1 );
}

// ---- Task: last block ----
// Regions[0]: q block M-1    (READ_ONLY)
// Regions[1]: q block M-2    (READ_ONLY)  -- to get last element as left boundary
// Regions[2]: dpdt block M-1 (WRITE_DISCARD)
inline void last_block_task( const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime )
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());
    Rect<1> prev_rect = runtime->get_index_space_domain(
        regions[1].get_logical_region().get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    coord_t prev_hi = prev_rect.hi[0];

    double q_l = q_prev[prev_hi];

    double coupling_lr = -signed_pow( q[lo] - q_l , LAMBDA - 1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[i] , KAPPA - 1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i + 1] , LAMBDA - 1 );
        dpdt[i] = val - coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA - 1 )
        + coupling_lr - signed_pow( q[hi] , LAMBDA - 1 );
}

// ---- Launch the oscillator chain force computation (non-blocking) ----
inline void osc_chain( const state_type &q , const state_type &dpdt ,
                       Context ctx , Runtime *runtime )
{
    const size_t M = q.num_blocks;

    // Retrieve sub-regions for q and dpdt
    std::vector<LogicalRegion> q_subs(M), dpdt_subs(M);
    for( size_t i = 0 ; i < M ; ++i )
    {
        q_subs[i] = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));
        dpdt_subs[i] = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)i)));
    }

    // First block
    {
        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[0], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[1], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(dpdt_subs[0], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        runtime->execute_task(ctx, launcher);
    }

    // Middle blocks
    for( size_t i = 1 ; i < M - 1 ; ++i )
    {
        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[i], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[i - 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(q_subs[i + 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        RegionRequirement rr3(dpdt_subs[i], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr3.add_field(FID_VAL);
        launcher.add_region_requirement(rr3);

        runtime->execute_task(ctx, launcher);
    }

    // Last block
    if( M >= 2 )
    {
        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[M - 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[M - 2], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(dpdt_subs[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        runtime->execute_task(ctx, launcher);
    }
}

// ---- Launch with global barrier (waits for all block tasks to complete) ----
inline void osc_chain_gb( const state_type &q , const state_type &dpdt ,
                          Context ctx , Runtime *runtime )
{
    const size_t M = q.num_blocks;
    std::vector<Future> futures;

    std::vector<LogicalRegion> q_subs(M), dpdt_subs(M);
    for( size_t i = 0 ; i < M ; ++i )
    {
        q_subs[i] = runtime->get_logical_subregion_by_color(
            q.lp, DomainPoint(Point<1>((coord_t)i)));
        dpdt_subs[i] = runtime->get_logical_subregion_by_color(
            dpdt.lp, DomainPoint(Point<1>((coord_t)i)));
    }

    // First block
    {
        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[0], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[1], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(dpdt_subs[0], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Middle blocks
    for( size_t i = 1 ; i < M - 1 ; ++i )
    {
        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[i], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[i - 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(q_subs[i + 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        RegionRequirement rr3(dpdt_subs[i], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr3.add_field(FID_VAL);
        launcher.add_region_requirement(rr3);

        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Last block
    if( M >= 2 )
    {
        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));

        RegionRequirement rr0(q_subs[M - 1], READ_ONLY, EXCLUSIVE, q.lr);
        rr0.add_field(FID_VAL);
        launcher.add_region_requirement(rr0);

        RegionRequirement rr1(q_subs[M - 2], READ_ONLY, EXCLUSIVE, q.lr);
        rr1.add_field(FID_VAL);
        launcher.add_region_requirement(rr1);

        RegionRequirement rr2(dpdt_subs[M - 1], WRITE_DISCARD, EXCLUSIVE, dpdt.lr);
        rr2.add_field(FID_VAL);
        launcher.add_region_requirement(rr2);

        futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier
    for( auto &f : futures )
        f.get_void_result();
}

// ---- Energy computation on raw vectors (unchanged from original) ----
inline double energy( const std::vector<double> &q , const std::vector<double> &p )
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double result = 0.5 * pow( abs(q[0]) , LAMBDA ) / LAMBDA;
    for( size_t i = 0 ; i < N - 1 ; ++i )
    {
        result += 0.5 * p[i] * p[i] + pow( q[i] , KAPPA ) / KAPPA
            + pow( abs(q[i] - q[i + 1]) , LAMBDA ) / LAMBDA;
    }
    result += 0.5 * p[N - 1] * p[N - 1] + pow( q[N - 1] , KAPPA ) / KAPPA
        + 0.5 * pow( abs(q[N - 1]) , LAMBDA ) / LAMBDA;
    return result;
}

// ---- Energy computation on Legion state_type (inline-maps the regions) ----
inline double energy( const state_type &q_state , const state_type &p_state ,
                      Context ctx , Runtime *runtime )
{
    // Inline-map the full q and p regions (forces pending writes to complete)
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_req);

    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_req);

    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        q_state.lr.get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    using checked_math::pow;
    using std::abs;

    double result = 0.5 * pow( abs((double)q_acc[lo]) , LAMBDA ) / LAMBDA;
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double qi  = q_acc[i];
        double qi1 = q_acc[i + 1];
        double pi  = p_acc[i];
        result += 0.5 * pi * pi + pow( qi , KAPPA ) / KAPPA
            + pow( abs(qi - qi1) , LAMBDA ) / LAMBDA;
    }
    {
        double qhi = q_acc[hi];
        double phi = p_acc[hi];
        result += 0.5 * phi * phi + pow( qhi , KAPPA ) / KAPPA
            + 0.5 * pow( abs(qhi) , LAMBDA ) / LAMBDA;
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return result;
}

// ---- Register all system tasks with the Legion runtime ----
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(FIRST_BLOCK_TASK_ID, "first_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<first_block_task>(
            registrar, "first_block");
    }
    {
        TaskVariantRegistrar registrar(CENTER_BLOCK_TASK_ID, "center_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<center_block_task>(
            registrar, "center_block");
    }
    {
        TaskVariantRegistrar registrar(LAST_BLOCK_TASK_ID, "last_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<last_block_task>(
            registrar, "last_block");
    }
}

#endif
