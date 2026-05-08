// Copyright 2013 Mario Mulansky

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

#include "shared_resize.hpp"

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

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
    using boost::math::sign;
    using std::abs;
    return checked_math::pow( x , k ) * sign(x);
}

enum SystemTaskIDs {
    FIRST_BLOCK_TASK_ID  = 50,
    CENTER_BLOCK_TASK_ID = 51,
    LAST_BLOCK_TASK_ID   = 52,
};

// ----------------------------------------------------------------
// First block task
//   regions[0]: q block 0       (READ_ONLY)
//   regions[1]: q block 1       (READ_ONLY)  -- need first element
//   regions[2]: dpdt block 0    (WRITE_DISCARD)
// ----------------------------------------------------------------
inline void first_block_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> right_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    double q_r = q_next[Point<1>(right_rect.lo[0])];

    double coupling_lr = -signed_pow( q[Point<1>(lo)] , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        val -= coupling_lr;
        dpdt[Point<1>(i)] = val;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] - q_r , LAMBDA-1 );
}

// ----------------------------------------------------------------
// Center block task
//   regions[0]: q block i       (READ_ONLY)
//   regions[1]: q block i-1     (READ_ONLY)  -- need last element
//   regions[2]: q block i+1     (READ_ONLY)  -- need first element
//   regions[3]: dpdt block i    (WRITE_DISCARD)
// ----------------------------------------------------------------
inline void center_block_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[3], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> left_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));
    Rect<1> right_rect(runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    double q_l = q_prev[Point<1>(left_rect.hi[0])];
    double q_r = q_next[Point<1>(right_rect.lo[0])];

    double coupling_lr = -signed_pow( q[Point<1>(lo)] - q_l , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        val -= coupling_lr;
        dpdt[Point<1>(i)] = val;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] - q_r , LAMBDA-1 );
}

// ----------------------------------------------------------------
// Last block task
//   regions[0]: q block M-1     (READ_ONLY)
//   regions[1]: q block M-2     (READ_ONLY)  -- need last element
//   regions[2]: dpdt block M-1  (WRITE_DISCARD)
// ----------------------------------------------------------------
inline void last_block_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[2], FID_VAL);

    Rect<1> rect(runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space()));
    Rect<1> left_rect(runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space()));

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    double q_l = q_prev[Point<1>(left_rect.hi[0])];

    double coupling_lr = -signed_pow( q[Point<1>(lo)] - q_l , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        double val = -signed_pow( q[Point<1>(i)] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[Point<1>(i)] - q[Point<1>(i+1)] , LAMBDA-1 );
        val -= coupling_lr;
        dpdt[Point<1>(i)] = val;
    }
    dpdt[Point<1>(hi)] = -signed_pow( q[Point<1>(hi)] , KAPPA-1 )
        + coupling_lr - signed_pow( q[Point<1>(hi)] , LAMBDA-1 );
}

// ----------------------------------------------------------------
// Helper: launch all oscillator-chain tasks (non-blocking).
// Optionally collects futures when barrier is needed.
// ----------------------------------------------------------------
inline void launch_osc_chain_tasks( state_type &q , state_type &dpdt ,
                                    Context ctx , Runtime *runtime ,
                                    std::vector<Future> *futures = nullptr )
{
    const int M = static_cast<int>( q.num_partitions );

    // ---------- first block ----------
    {
        TaskLauncher launcher( FIRST_BLOCK_TASK_ID , TaskArgument(NULL,0) );

        LogicalRegion q0 = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(0)) );
        LogicalRegion q1 = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(1)) );
        LogicalRegion d0 = runtime->get_logical_subregion_by_color(ctx, dpdt.lp , DomainPoint(Point<1>(0)) );

        launcher.add_region_requirement(
            RegionRequirement( q0 , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[0].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( q1 , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[1].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( d0 , WRITE_DISCARD , EXCLUSIVE , dpdt.lr ) );
        launcher.region_requirements[2].add_field( FID_VAL );

        Future f = runtime->execute_task( ctx , launcher );
        if( futures ) futures->push_back( f );
    }

    // ---------- center blocks ----------
    for( int i = 1 ; i < M-1 ; ++i )
    {
        TaskLauncher launcher( CENTER_BLOCK_TASK_ID , TaskArgument(NULL,0) );

        LogicalRegion qi      = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(i)) );
        LogicalRegion qi_prev = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(i-1)) );
        LogicalRegion qi_next = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(i+1)) );
        LogicalRegion di      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp , DomainPoint(Point<1>(i)) );

        launcher.add_region_requirement(
            RegionRequirement( qi , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[0].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( qi_prev , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[1].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( qi_next , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[2].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( di , WRITE_DISCARD , EXCLUSIVE , dpdt.lr ) );
        launcher.region_requirements[3].add_field( FID_VAL );

        Future f = runtime->execute_task( ctx , launcher );
        if( futures ) futures->push_back( f );
    }

    // ---------- last block ----------
    if( M > 1 )
    {
        TaskLauncher launcher( LAST_BLOCK_TASK_ID , TaskArgument(NULL,0) );

        LogicalRegion qN      = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(M-1)) );
        LogicalRegion qN_prev = runtime->get_logical_subregion_by_color(ctx, q.lp , DomainPoint(Point<1>(M-2)) );
        LogicalRegion dN      = runtime->get_logical_subregion_by_color(ctx, dpdt.lp , DomainPoint(Point<1>(M-1)) );

        launcher.add_region_requirement(
            RegionRequirement( qN , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[0].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( qN_prev , READ_ONLY , EXCLUSIVE , q.lr ) );
        launcher.region_requirements[1].add_field( FID_VAL );

        launcher.add_region_requirement(
            RegionRequirement( dN , WRITE_DISCARD , EXCLUSIVE , dpdt.lr ) );
        launcher.region_requirements[2].add_field( FID_VAL );

        Future f = runtime->execute_task( ctx , launcher );
        if( futures ) futures->push_back( f );
    }
}

// ----------------------------------------------------------------
// osc_chain: non-blocking task launch (analogous to the HPX
//            dataflow version without global barrier)
// ----------------------------------------------------------------
struct osc_chain
{
    Context  ctx;
    Runtime *runtime;

    osc_chain( Context c , Runtime *r ) : ctx(c) , runtime(r) {}

    void operator()( state_type &q , state_type &dpdt ) const
    {
        launch_osc_chain_tasks( q , dpdt , ctx , runtime );
    }
};

// ----------------------------------------------------------------
// osc_chain_gb: same but with a global barrier at the end
// ----------------------------------------------------------------
struct osc_chain_gb
{
    Context  ctx;
    Runtime *runtime;

    osc_chain_gb( Context c , Runtime *r ) : ctx(c) , runtime(r) {}

    void operator()( state_type &q , state_type &dpdt ) const
    {
        std::vector<Future> futures;
        launch_osc_chain_tasks( q , dpdt , ctx , runtime , &futures );
        for( auto &f : futures )
            f.get_void_result();
    }
};

// ----------------------------------------------------------------
// Energy computation on plain vectors
// ----------------------------------------------------------------
inline double energy( const std::vector<double> &q , const std::vector<double> &p )
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double en = 0.5*pow( abs(q[0]) , LAMBDA ) / LAMBDA;
    for( size_t i=0 ; i<N-1 ; ++i )
    {
        en += 0.5*p[i]*p[i] + pow( q[i] , KAPPA ) / KAPPA
            + pow( abs(q[i]-q[i+1]) , LAMBDA ) / LAMBDA;
    }
    en += 0.5*p[N-1]*p[N-1] + pow( q[N-1] , KAPPA ) / KAPPA
        + 0.5*pow( abs(q[N-1]) , LAMBDA ) / LAMBDA;
    return en;
}

// ----------------------------------------------------------------
// Energy computation on Legion state_type (inline-maps regions)
// ----------------------------------------------------------------
inline double energy( const state_type &q_state , const state_type &p_state ,
                      Context ctx , Runtime *runtime )
{
    const size_t N = q_state.num_partitions * q_state.partition_size;

    // Inline-map the full q region
    RegionRequirement q_req( q_state.lr , READ_ONLY , EXCLUSIVE , q_state.lr );
    q_req.add_field( FID_VAL );
    InlineLauncher q_il( q_req );
    PhysicalRegion q_phys = runtime->map_region( ctx , q_il );

    // Inline-map the full p region
    RegionRequirement p_req( p_state.lr , READ_ONLY , EXCLUSIVE , p_state.lr );
    p_req.add_field( FID_VAL );
    InlineLauncher p_il( p_req );
    PhysicalRegion p_phys = runtime->map_region( ctx , p_il );

    q_phys.wait_until_valid();
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc( q_phys , FID_VAL );
    const FieldAccessor<READ_ONLY, double, 1> p_acc( p_phys , FID_VAL );

    // Copy into contiguous vectors for the scalar energy routine
    std::vector<double> q_vec( N ) , p_vec( N );
    for( size_t i = 0 ; i < N ; ++i )
    {
        q_vec[i] = q_acc[ Point<1>( static_cast<coord_t>(i) ) ];
        p_vec[i] = p_acc[ Point<1>( static_cast<coord_t>(i) ) ];
    }

    runtime->unmap_region( ctx , q_phys );
    runtime->unmap_region( ctx , p_phys );

    return energy( q_vec , p_vec );
}

// ----------------------------------------------------------------
// Task pre-registration (call before Runtime::start)
// ----------------------------------------------------------------
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar( FIRST_BLOCK_TASK_ID , "first_block_task" );
        registrar.add_constraint( ProcessorConstraint( Processor::LOC_PROC ) );
        Runtime::preregister_task_variant<first_block_task>(
            registrar , "first_block_task" );
    }
    {
        TaskVariantRegistrar registrar( CENTER_BLOCK_TASK_ID , "center_block_task" );
        registrar.add_constraint( ProcessorConstraint( Processor::LOC_PROC ) );
        Runtime::preregister_task_variant<center_block_task>(
            registrar , "center_block_task" );
    }
    {
        TaskVariantRegistrar registrar( LAST_BLOCK_TASK_ID , "last_block_task" );
        registrar.add_constraint( ProcessorConstraint( Processor::LOC_PROC ) );
        Runtime::preregister_task_variant<last_block_task>(
            registrar , "last_block_task" );
    }
}

#endif
