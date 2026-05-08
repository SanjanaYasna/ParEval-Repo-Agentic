// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Defined in odeint.cpp
extern const TaskID FOR_EACH3_TASK_ID;
extern const FieldID FID_VAL;
extern Runtime *legion_runtime_g;
extern Context legion_context_g;

// Argument struct passed to the FOR_EACH3 index task.
// Encodes the scale_sum2 coefficients and which region requirement
// index corresponds to each of the three operands (to handle aliasing).
struct ForEach3Args {
    double alpha1;
    double alpha2;
    int s2_region_idx;   // region requirement index for the s2 operand
    int s3_region_idx;   // region requirement index for the s3 operand
};

struct local_dataflow_algebra
{
    // template< typename S , typename Op >
    // for now just a single state type
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        const size_t N = s1.size();
        if( N == 0 ) return;

        // Detect aliasing: odeint may pass the same state object for
        // both s1 and s2 (e.g. q = 1*q + dt*dpdt).
        bool s1_eq_s2 = ( &s1 == &s2 );
        bool s1_eq_s3 = ( &s1 == &s3 );
        bool s2_eq_s3 = ( &s2 == &s3 );

        // Build the task argument with coefficients and region mapping
        ForEach3Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;

        // Region 0 is always s1 (READ_WRITE).
        // Assign region indices for s2 and s3, merging when aliased.
        int num_reqs = 1;

        args.s2_region_idx = s1_eq_s2 ? 0 : num_reqs++;

        if( s1_eq_s3 )
            args.s3_region_idx = 0;
        else if( s2_eq_s3 )
            args.s3_region_idx = args.s2_region_idx;
        else
            args.s3_region_idx = num_reqs++;

        // Launch one point task per block (partition color).
        // Legion resolves data dependencies through region requirements.
        IndexLauncher launcher( FOR_EACH3_TASK_ID ,
                                s1.color_space ,
                                TaskArgument( &args , sizeof(args) ) ,
                                ArgumentMap() );

        // Region 0: s1 – the destination (read-write)
        launcher.add_region_requirement(
            RegionRequirement( s1.partition , 0 /*identity projection*/ ,
                               READ_WRITE , EXCLUSIVE , s1.parent ) );
        launcher.region_requirements[0].add_field( FID_VAL );

        // s2 region (read-only, added only when not aliased with s1)
        if( !s1_eq_s2 )
        {
            launcher.add_region_requirement(
                RegionRequirement( s2.partition , 0 /*identity projection*/ ,
                                   READ_ONLY , EXCLUSIVE , s2.parent ) );
            launcher.region_requirements[args.s2_region_idx].add_field( FID_VAL );
        }

        // s3 region (read-only, added only when not aliased with s1 or s2)
        if( !s1_eq_s3 && !s2_eq_s3 )
        {
            launcher.add_region_requirement(
                RegionRequirement( s3.partition , 0 /*identity projection*/ ,
                                   READ_ONLY , EXCLUSIVE , s3.parent ) );
            launcher.region_requirements[args.s3_region_idx].add_field( FID_VAL );
        }

        // Non-blocking launch – Legion sequences subsequent tasks that
        // touch the same regions, mirroring HPX dataflow chaining.
        legion_runtime_g->execute_index_space( legion_context_g , launcher );
    }
};

#endif
