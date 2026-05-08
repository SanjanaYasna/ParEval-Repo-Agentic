// Copyright 2013 Mario Mulansky
// Translated to Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Task IDs shared across the application
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
};

// Field IDs shared across the application
enum {
    FID_VAL = 0,
};

// Serializable arguments for the scale_sum2 index task
struct ScaleSumArgs {
    double alpha1;
    double alpha2;
    bool same_s1_s2; // true when s1 and s2 refer to the same logical region
};

struct local_dataflow_algebra
{
    // for_each3: applies op element-wise across three state containers.
    // S is expected to be state_type with members:
    //   runtime, ctx, color_is, lp, lr
    // Op is expected to have members: m_alpha1, m_alpha2 (e.g. scale_sum2)
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        // Detect aliasing: in odeint's symplectic stepper, s1 == s2 is common
        // (e.g. q = 1.0*q + c*dt*p). We must avoid conflicting region
        // requirements on the same logical region within one launch.
        bool same12 = (s1.lr == s2.lr);

        ScaleSumArgs args;
        args.alpha1    = static_cast<double>(op.m_alpha1);
        args.alpha2    = static_cast<double>(op.m_alpha2);
        args.same_s1_s2 = same12;

        // Launch one point task per partition (block), all in parallel
        IndexTaskLauncher launcher(SCALE_SUM2_TASK_ID,
                                   s1.color_is,
                                   TaskArgument(&args, sizeof(args)),
                                   ArgumentMap());

        // Region requirement 0: s1 — READ_WRITE (destination; also source
        // when aliased with s2)
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        if (same12) {
            // s1 and s2 share the same logical region.
            // The point task will read s2 data through the READ_WRITE
            // accessor on region 0.
            //
            // Region requirement 1: s3 (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[1].add_field(FID_VAL);
        } else {
            // All three states are backed by distinct logical regions.
            //
            // Region requirement 1: s2 (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(FID_VAL);

            // Region requirement 2: s3 (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[2].add_field(FID_VAL);
        }

        // Launch asynchronously — Legion's dependence analysis on the
        // logical regions enforces correct ordering with subsequent
        // task launches (mirroring HPX's future-chain semantics).
        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
