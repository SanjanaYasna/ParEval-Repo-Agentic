// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <legion.h>
#include <boost/math/special_functions/sign.hpp>

#include "shared_resize.hpp"

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum SystemTaskIDs {
    GET_FIRST_ELEMENT_TASK_ID = 20,
    GET_LAST_ELEMENT_TASK_ID  = 21,
    SYSTEM_BLOCK_TASK_ID      = 22,
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

// ---- Task argument struct ----

struct SystemBlockArgs {
    size_t block_index;
    size_t num_blocks;
};

// ---- Leaf task: extract first element of a subregion ----

double get_first_element_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.lo];
}

// ---- Leaf task: extract last element of a subregion ----

double get_last_element_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.hi];
}

// ---- Main computation task: compute dpdt for one block ----
// Region 0: q   (this block, READ_ONLY)
// Region 1: dpdt (this block, WRITE_DISCARD)
// Futures (in order): q_l (last elem of prev block, if exists),
//                     q_r (first elem of next block, if exists)

void system_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    SystemBlockArgs args = *(const SystemBlockArgs *)task->args;
    size_t block_idx  = args.block_index;
    size_t num_blocks = args.num_blocks;

    const FieldAccessor<READ_ONLY, double, 1>    q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    // Copy q into a local buffer
    std::vector<double> q(N);
    for( coord_t j = lo ; j <= hi ; ++j )
        q[j - lo] = q_acc[j];

    // Determine boundary conditions
    bool has_left  = (block_idx > 0);
    bool has_right = (block_idx < num_blocks - 1);

    size_t future_idx = 0;
    double q_l = 0.0, q_r = 0.0;
    if( has_left )
        q_l = task->futures[future_idx++].get_result<double>();
    if( has_right )
        q_r = task->futures[future_idx++].get_result<double>();

    // Compute dpdt
    std::vector<double> dpdt(N);

    // Left boundary coupling
    double coupling_lr;
    if( has_left )
        coupling_lr = -signed_pow( q[0] - q_l , LAMBDA-1 );
    else
        coupling_lr = -signed_pow( q[0] , LAMBDA-1 );

    // Interior
    for( size_t i = 0 ; i < N-1 ; ++i )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }

    // Right boundary coupling
    if( has_right )
        dpdt[N-1] = -signed_pow( q[N-1] , KAPPA-1 )
            + coupling_lr - signed_pow( q[N-1] - q_r , LAMBDA-1 );
    else
        dpdt[N-1] = -signed_pow( q[N-1] , KAPPA-1 )
            + coupling_lr - signed_pow( q[N-1] , LAMBDA-1 );

    // Write dpdt back to region
    for( coord_t j = lo ; j <= hi ; ++j )
        dpdt_acc[j] = dpdt[j - lo];
}

// ---- osc_chain: launch system tasks (no global barrier) ----

void osc_chain( state_type &q , state_type &dpdt )
{
    Runtime *runtime = LegionHelper::runtime;
    Context  ctx     = LegionHelper::ctx;
    const size_t M   = q.M;

    // Phase 1: extract boundary values as lightweight futures
    std::vector<Future> first_elem(M);
    std::vector<Future> last_elem(M);

    for( size_t i = 0 ; i < M ; ++i )
    {
        LogicalRegion q_sub = q.subregion(i);
        {
            TaskLauncher launcher(GET_FIRST_ELEMENT_TASK_ID,
                                  TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY,
                                  EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            first_elem[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(GET_LAST_ELEMENT_TASK_ID,
                                  TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY,
                                  EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            last_elem[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // Phase 2: launch computation tasks with boundary futures
    for( size_t i = 0 ; i < M ; ++i )
    {
        LogicalRegion q_sub = q.subregion(i);
        LogicalRegion dpdt_sub = dpdt.subregion(i);

        SystemBlockArgs args;
        args.block_index = i;
        args.num_blocks  = M;

        TaskLauncher launcher(SYSTEM_BLOCK_TASK_ID,
                              TaskArgument(&args, sizeof(args)));

        // Region 0: q for this block (READ)
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY,
                              EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        // Region 1: dpdt for this block (WRITE)
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD,
                              EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        // Add boundary futures (order matters: q_l first, then q_r)
        if( i > 0 )
            launcher.add_future(last_elem[i - 1]);
        if( i < M - 1 )
            launcher.add_future(first_elem[i + 1]);

        runtime->execute_task(ctx, launcher);
    }
}

// ---- osc_chain_gb: same as osc_chain but with global barrier ----

void osc_chain_gb( state_type &q , state_type &dpdt )
{
    Runtime *runtime = LegionHelper::runtime;
    Context  ctx     = LegionHelper::ctx;
    const size_t M   = q.M;

    std::vector<Future> first_elem(M);
    std::vector<Future> last_elem(M);

    for( size_t i = 0 ; i < M ; ++i )
    {
        LogicalRegion q_sub = q.subregion(i);
        {
            TaskLauncher launcher(GET_FIRST_ELEMENT_TASK_ID,
                                  TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY,
                                  EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            first_elem[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(GET_LAST_ELEMENT_TASK_ID,
                                  TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY,
                                  EXCLUSIVE, q.lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            last_elem[i] = runtime->execute_task(ctx, launcher);
        }
    }

    std::vector<Future> compute_futures;
    for( size_t i = 0 ; i < M ; ++i )
    {
        LogicalRegion q_sub = q.subregion(i);
        LogicalRegion dpdt_sub = dpdt.subregion(i);

        SystemBlockArgs args;
        args.block_index = i;
        args.num_blocks  = M;

        TaskLauncher launcher(SYSTEM_BLOCK_TASK_ID,
                              TaskArgument(&args, sizeof(args)));

        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY,
                              EXCLUSIVE, q.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD,
                              EXCLUSIVE, dpdt.lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        if( i > 0 )
            launcher.add_future(last_elem[i - 1]);
        if( i < M - 1 )
            launcher.add_future(first_elem[i + 1]);

        compute_futures.push_back(runtime->execute_task(ctx, launcher));
    }

    // Global barrier
    for( auto &f : compute_futures )
        f.get_void_result();
}

// ---- Energy computation (raw vectors) ----

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

// ---- Energy computation (state_type – uses inline mapping) ----

double energy( const state_type &q_state , const state_type &p_state )
{
    Runtime *runtime = LegionHelper::runtime;
    Context  ctx     = LegionHelper::ctx;

    dvec q, p;

    for( size_t i = 0 ; i < q_state.M ; ++i )
    {
        LogicalRegion q_sub = q_state.subregion(i);
        LogicalRegion p_sub = p_state.subregion(i);

        // Map q subregion inline
        {
            RegionRequirement req(q_sub, READ_ONLY,
                                  EXCLUSIVE, q_state.lr);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VAL);
            Rect<1> rect = runtime->get_index_space_domain(
                q_sub.get_index_space());

            for( PointInRectIterator<1> itr(rect) ; itr() ; itr++ )
                q.push_back(acc[*itr]);

            runtime->unmap_region(ctx, pr);
        }

        // Map p subregion inline
        {
            RegionRequirement req(p_sub, READ_ONLY,
                                  EXCLUSIVE, p_state.lr);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VAL);
            Rect<1> rect = runtime->get_index_space_domain(
                p_sub.get_index_space());

            for( PointInRectIterator<1> itr(rect) ; itr() ; itr++ )
                p.push_back(acc[*itr]);

            runtime->unmap_region(ctx, pr);
        }
    }

    return energy( q , p );
}

// ---- Task registration (call before Runtime::start) ----

void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(GET_FIRST_ELEMENT_TASK_ID,
                                       "get_first_element");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, get_first_element_task>(
            registrar, "get_first_element");
    }
    {
        TaskVariantRegistrar registrar(GET_LAST_ELEMENT_TASK_ID,
                                       "get_last_element");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, get_last_element_task>(
            registrar, "get_last_element");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID,
                                       "system_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_block_task>(
            registrar, "system_block");
    }
}

#endif
