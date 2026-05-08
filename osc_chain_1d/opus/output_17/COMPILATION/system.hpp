// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include "legion.h"

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Field ID shared across the application
enum FieldIDs {
    FID_VAL = 101,
};

// Task IDs for system tasks
enum SystemTaskIDs {
    TASK_COMPUTE_BLOCK = 10,
};

// Legion state representation: a partitioned logical region
struct state_type {
    LogicalRegion lr;          // parent logical region with N elements
    LogicalPartition lp;       // equal partition into M blocks of G elements
    size_t M;                  // number of blocks
    size_t G;                  // elements per block
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
    using std::abs;
    double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow( x , k ) * s;
}

// Task argument for the block computation task
struct SystemBlockArgs {
    int block_index;
    int num_blocks;
    int block_size;
};

// Legion task: compute dpdt for a single block of the oscillator chain
// Region layout:
//   0: q[idx]    (READ_ONLY)
//   1: dpdt[idx] (WRITE_DISCARD)
//   2: q[idx-1]  (READ_ONLY)  -- only if idx > 0
//   next: q[idx+1] (READ_ONLY) -- only if idx < M-1
inline void compute_block_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
    SystemBlockArgs args = *(const SystemBlockArgs *)task->args;
    int idx = args.block_index;
    int M   = args.num_blocks;
    int G   = args.block_size;

    bool is_first = (idx == 0);
    bool is_last  = (idx == M - 1);

    // Accessors for own q and dpdt subregions
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Domain q_dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> q_rect = q_dom;
    coord_t q_base = q_rect.lo[0];

    Domain dpdt_dom = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());
    Rect<1> dpdt_rect = dpdt_dom;
    coord_t dpdt_base = dpdt_rect.lo[0];

    // Copy q block into a local buffer for computation
    std::vector<double> q(G);
    for( int i = 0 ; i < G ; i++ )
        q[i] = q_acc[Point<1>(q_base + i)];

    // Read boundary values from neighbor subregions
    double q_l = 0.0;
    double q_r = 0.0;
    int reg_idx = 2;

    if( !is_first )
    {
        const FieldAccessor<READ_ONLY, double, 1> left_acc(regions[reg_idx], FID_VAL);
        Domain left_dom = runtime->get_index_space_domain(
            task->regions[reg_idx].region.get_index_space());
        Rect<1> left_rect = left_dom;
        q_l = left_acc[Point<1>(left_rect.hi[0])]; // last element of left neighbor
        reg_idx++;
    }
    if( !is_last )
    {
        const FieldAccessor<READ_ONLY, double, 1> right_acc(regions[reg_idx], FID_VAL);
        Domain right_dom = runtime->get_index_space_domain(
            task->regions[reg_idx].region.get_index_space());
        Rect<1> right_rect = right_dom;
        q_r = right_acc[Point<1>(right_rect.lo[0])]; // first element of right neighbor
    }

    // Compute dpdt for this block
    std::vector<double> dpdt(G);

    if( is_first )
    {
        // First block: left boundary couples to fixed q=0
        double coupling_lr = -signed_pow( q[0] , LAMBDA-1 );
        for( int i = 0 ; i < G-1 ; i++ )
        {
            dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
            coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
            dpdt[i] -= coupling_lr;
        }
        dpdt[G-1] = -signed_pow( q[G-1] , KAPPA-1 )
            + coupling_lr - signed_pow( q[G-1] - q_r , LAMBDA-1 );
    }
    else if( is_last )
    {
        // Last block: right boundary couples to fixed q=0
        double coupling_lr = -signed_pow( q[0] - q_l , LAMBDA-1 );
        for( int i = 0 ; i < G-1 ; i++ )
        {
            dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
            coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
            dpdt[i] -= coupling_lr;
        }
        dpdt[G-1] = -signed_pow( q[G-1] , KAPPA-1 )
            + coupling_lr - signed_pow( q[G-1] , LAMBDA-1 );
    }
    else
    {
        // Interior block
        double coupling_lr = -signed_pow( q[0] - q_l , LAMBDA-1 );
        for( int i = 0 ; i < G-1 ; i++ )
        {
            dpdt[i] = -signed_pow( q[i] , KAPPA-1 ) + coupling_lr;
            coupling_lr = signed_pow( q[i] - q[i+1] , LAMBDA-1 );
            dpdt[i] -= coupling_lr;
        }
        dpdt[G-1] = -signed_pow( q[G-1] , KAPPA-1 )
            + coupling_lr - signed_pow( q[G-1] - q_r , LAMBDA-1 );
    }

    // Write computed dpdt back to the region
    for( int i = 0 ; i < G ; i++ )
        dpdt_acc[Point<1>(dpdt_base + i)] = dpdt[i];
}

// Launch oscillator chain force computation as Legion tasks
// Each block is a separate task; dependencies via region requirements allow
// non-adjacent blocks to execute in parallel (analogous to HPX dataflow).
inline void osc_chain(state_type &q, state_type &dpdt,
                      Runtime *runtime, Context ctx)
{
    const size_t M = q.M;
    const size_t G = q.G;

    for( size_t i = 0 ; i < M ; i++ )
    {
        SystemBlockArgs args;
        args.block_index = static_cast<int>(i);
        args.num_blocks  = static_cast<int>(M);
        args.block_size  = static_cast<int>(G);

        TaskLauncher launcher(TASK_COMPUTE_BLOCK,
                              TaskArgument(&args, sizeof(SystemBlockArgs)));

        // Region 0: q[i] (READ_ONLY)
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(
            q.lp, static_cast<Color>(i));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q.lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 1: dpdt[i] (WRITE_DISCARD)
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(
            dpdt.lp, static_cast<Color>(i));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 2: q[i-1] (READ_ONLY) -- left neighbor boundary
        if( i > 0 )
        {
            LogicalRegion q_left = runtime->get_logical_subregion_by_color(
                q.lp, static_cast<Color>(i - 1));
            launcher.add_region_requirement(
                RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        // Region 2 or 3: q[i+1] (READ_ONLY) -- right neighbor boundary
        if( i < M - 1 )
        {
            LogicalRegion q_right = runtime->get_logical_subregion_by_color(
                q.lp, static_cast<Color>(i + 1));
            launcher.add_region_requirement(
                RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, q.lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        // Non-blocking launch; Legion runtime resolves parallelism
        runtime->execute_task(ctx, launcher);
    }
}

// Launch with global barrier -- all tasks complete before returning
inline void osc_chain_gb(state_type &q, state_type &dpdt,
                         Runtime *runtime, Context ctx)
{
    osc_chain(q, dpdt, runtime, ctx);
    // Execution fence ensures all previously launched tasks complete
    runtime->issue_execution_fence(ctx).get_void_result();
}

// Compute total energy from flat vectors (helper)
inline double energy( const std::vector<double> &q , const std::vector<double> &p )
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double energy = 0.5*pow( abs(q[0]) , LAMBDA ) / LAMBDA;
    for( size_t i = 0 ; i < N-1 ; ++i )
    {
        energy += 0.5*p[i]*p[i] + pow( q[i] , KAPPA ) / KAPPA
            + pow( abs(q[i]-q[i+1]) , LAMBDA ) / LAMBDA;
    }
    energy += 0.5*p[N-1]*p[N-1] + pow( q[N-1] , KAPPA ) / KAPPA
        + 0.5*pow( abs(q[N-1]) , LAMBDA ) / LAMBDA;
    return energy;
}

// Compute total energy from Legion state_type (inline maps regions)
inline double energy(const state_type &q_state, const state_type &p_state,
                     Runtime *runtime, Context ctx)
{
    // Inline map the full q and p parent regions for reading
    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_req);

    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_req);

    q_phys.wait_until_valid();
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_phys, FID_VAL);

    const size_t N = q_state.M * q_state.G;

    // Gather into flat vectors for energy computation
    std::vector<double> q_vec(N), p_vec(N);
    for( size_t i = 0 ; i < N ; i++ )
    {
        q_vec[i] = q_acc[Point<1>(i)];
        p_vec[i] = p_acc[Point<1>(i)];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q_vec, p_vec);
}

// Pre-register all system tasks (call before Runtime::start)
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(TASK_COMPUTE_BLOCK, "compute_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compute_block_task>(
            registrar, "compute_block");
    }
}

#endif
