// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include "legion.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Field ID shared across the project (matches shared_resize.hpp, algebra.hpp, etc.)
#ifndef FID_VAL
#define FID_VAL 101
#endif

// Task IDs for system computation tasks
enum SystemTaskID {
    TASK_FIRST_BLOCK  = 30,
    TASK_CENTER_BLOCK = 31,
    TASK_LAST_BLOCK   = 32,
};

// ---- State type for Legion ----
// Replaces: typedef std::vector< shared_future< shared_vec > > state_type;
// Each block is a LogicalRegion (subregion of lr via partition lp).
#ifndef STATE_TYPE_DEFINED
#define STATE_TYPE_DEFINED
struct state_type {
    LogicalRegion    lr;          // full logical region (parent)
    LogicalPartition lp;          // partition into M blocks
    std::vector<LogicalRegion> blocks; // cached subregions, one per block
    size_t num_blocks;            // M
    size_t block_size;            // G (elements per block)

    size_t size() const { return num_blocks; }
    LogicalRegion&       operator[](size_t i)       { return blocks[i]; }
    const LogicalRegion& operator[](size_t i) const { return blocks[i]; }
};
#endif

// ---- Checked math helpers (unchanged) ----
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
    return checked_math::pow( x , k ) * sign(x);
}

// ======================================================================
//  Legion task implementations
// ======================================================================

// First block of the chain.
// Region 0 : q  (this block)       – READ_ONLY
// Region 1 : q  (next block)       – READ_ONLY  (need first element = q_r)
// Region 2 : dpdt (this block)     – WRITE_DISCARD
void first_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
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
    double q_r = q_next[next_rect.lo];              // first element of next block

    double coupling_lr = -signed_pow( q[lo] , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA-1 )
        + coupling_lr - signed_pow( q[hi] - q_r , LAMBDA-1 );
}

// Center (middle) block of the chain.
// Region 0 : q  (this block)       – READ_ONLY
// Region 1 : q  (previous block)   – READ_ONLY  (need last element = q_l)
// Region 2 : q  (next block)       – READ_ONLY  (need first element = q_r)
// Region 3 : dpdt (this block)     – WRITE_DISCARD
void center_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
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
    double q_l = q_prev[prev_rect.hi];              // last element of previous block
    double q_r = q_next[next_rect.lo];              // first element of next block

    double coupling_lr = -signed_pow( q[lo] - q_l , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA-1 )
        + coupling_lr - signed_pow( q[hi] - q_r , LAMBDA-1 );
}

// Last block of the chain.
// Region 0 : q  (this block)       – READ_ONLY
// Region 1 : q  (previous block)   – READ_ONLY  (need last element = q_l)
// Region 2 : dpdt (this block)     – WRITE_DISCARD
void last_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
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
    double q_l = q_prev[prev_rect.hi];              // last element of previous block

    double coupling_lr = -signed_pow( q[lo] - q_l , LAMBDA-1 );
    for( coord_t i = lo ; i < hi ; ++i )
    {
        dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
        coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
        dpdt[i] -= coupling_lr;
    }
    dpdt[hi] = -signed_pow( q[hi] , KAPPA-1 )
        + coupling_lr - signed_pow( q[hi] , LAMBDA-1 );
}

// ======================================================================
//  Helper: submit all system tasks for one evaluation of the RHS
// ======================================================================
inline void launch_system_tasks(state_type &q, state_type &dpdt,
                                Runtime *runtime, Context ctx,
                                std::vector<Future> *futures = nullptr)
{
    const size_t M = q.num_blocks;
    assert(M >= 2);

    auto store = [&](Future f) { if (futures) futures->push_back(f); };

    // ---- first block ----
    {
        TaskLauncher launcher(TASK_FIRST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[0], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.blocks[0], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        store(runtime->execute_task(ctx, launcher));
    }

    // ---- middle blocks ----
    for( size_t i = 1 ; i < M-1 ; ++i )
    {
        TaskLauncher launcher(TASK_CENTER_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[i], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[i-1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[i+1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.blocks[i], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        store(runtime->execute_task(ctx, launcher));
    }

    // ---- last block ----
    {
        TaskLauncher launcher(TASK_LAST_BLOCK, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[M-1], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(q.blocks[M-2], READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt.blocks[M-1], WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        store(runtime->execute_task(ctx, launcher));
    }
}

// ======================================================================
//  osc_chain – callable system functor (replaces HPX free function)
//  Captures Runtime*/Context so it can be passed to the stepper as
//  sys(q, dpdt).  Tasks are submitted asynchronously; the Legion
//  runtime resolves dependencies through region requirements.
// ======================================================================
struct osc_chain
{
    Runtime *runtime;
    Context  ctx;

    osc_chain(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    void operator()( state_type &q , state_type &dpdt ) const
    {
        launch_system_tasks(q, dpdt, runtime, ctx);
    }
};

// ======================================================================
//  osc_chain_gb – same computation with a global barrier at the end
//  (mirrors the HPX wait_all(dpdt) variant)
// ======================================================================
struct osc_chain_gb
{
    Runtime *runtime;
    Context  ctx;

    osc_chain_gb(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    void operator()( state_type &q , state_type &dpdt ) const
    {
        std::vector<Future> futures;
        launch_system_tasks(q, dpdt, runtime, ctx, &futures);
        // global barrier – wait for every block task to finish
        for( auto &f : futures )
            f.get_void_result();
    }
};

// ======================================================================
//  Energy computation
// ======================================================================

// Energy from raw vectors (unchanged physics)
double energy( const std::vector<double> &q , const std::vector<double> &p )
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double e = 0.5*pow( abs(q[0]) , LAMBDA) / LAMBDA;
    for( size_t i=0 ; i<N-1 ; ++i )
    {
        e += 0.5*p[i]*p[i] + pow( q[i] , KAPPA ) / KAPPA
            + pow( abs(q[i]-q[i+1]) , LAMBDA ) / LAMBDA;
    }
    e += 0.5*p[N-1]*p[N-1] + pow( q[N-1] , KAPPA ) / KAPPA
        + 0.5*pow( abs(q[N-1]) , LAMBDA) / LAMBDA;
    return e;
}

// Energy from Legion state_type – inline-maps the full regions,
// copies into std::vectors, and delegates to the vector overload.
// (Replaces the HPX template that called .get() on every future.)
double energy( const state_type &q_state , const state_type &p_state ,
               Runtime *runtime , Context ctx )
{
    // Inline-map the full logical regions for reading
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);

    PhysicalRegion q_pr = runtime->map_region(ctx, q_req);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_req);
    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        q_state.lr.get_index_space());
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = static_cast<size_t>(hi - lo + 1);

    // Collect into contiguous vectors
    std::vector<double> q_vec(N), p_vec(N);
    for( coord_t i = lo ; i <= hi ; ++i )
    {
        q_vec[i - lo] = q_acc[i];
        p_vec[i - lo] = p_acc[i];
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return energy(q_vec, p_vec);
}

// ======================================================================
//  Task registration (call once before Runtime::start)
// ======================================================================
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(TASK_FIRST_BLOCK, "first_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<first_block_task>(
            registrar, "first_block_task");
    }
    {
        TaskVariantRegistrar registrar(TASK_CENTER_BLOCK, "center_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<center_block_task>(
            registrar, "center_block_task");
    }
    {
        TaskVariantRegistrar registrar(TASK_LAST_BLOCK, "last_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<last_block_task>(
            registrar, "last_block_task");
    }
}

#endif // SYSTEM_HPP
