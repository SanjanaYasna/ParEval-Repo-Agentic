// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Global Legion runtime context (set in top-level task, defined in odeint.cpp)
extern Runtime *global_runtime;
extern Context  global_ctx;

// Field and task IDs (defined in odeint.cpp)
extern FieldID FID_VAL;
extern TaskID  SCALE_SUM2_TASK_ID;

// Arguments passed to the scale_sum2 index task
struct scale_sum2_task_args_t {
    double alpha1;
    double alpha2;
    bool   s1_eq_s2;  // true if s1 and s2 share the same logical region
    bool   s1_eq_s3;  // true if s1 and s3 share the same logical region
};

struct local_dataflow_algebra
{
    // Applies op element-wise: s1[i] = op(s1[i], s2[i], s3[i]) for each block i.
    // In the HPX version this created chained dataflow futures; in Legion we
    // launch an index space of tasks and let the runtime resolve dependencies
    // through region usage.
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        // Build task arguments from the operation's coefficients
        scale_sum2_task_args_t args;
        args.alpha1   = op.m_alpha1;
        args.alpha2   = op.m_alpha2;
        args.s1_eq_s2 = (s1.lr == s2.lr);
        args.s1_eq_s3 = (s1.lr == s3.lr);

        // Create an index launcher over the M blocks (one point task per block)
        IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                               s1.color_space,
                               TaskArgument(&args, sizeof(args)),
                               ArgumentMap());

        // Region 0: s1 – always read-write (destination)
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.add_field(0, FID_VAL);

        // Region for s2 – read-only, added only when s2 is a distinct region
        // from s1.  When aliased the task reads s2 data from region 0.
        if (!args.s1_eq_s2)
        {
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.add_field(1, FID_VAL);
        }

        // Region for s3 – read-only, added only when s3 is a distinct region
        // from s1.  The requirement index shifts depending on whether s2 was
        // added above.
        if (!args.s1_eq_s3)
        {
            int reg_idx = args.s1_eq_s2 ? 1 : 2;
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.add_field(reg_idx, FID_VAL);
        }

        // Launch; Legion automatically orders this with prior and subsequent
        // tasks that touch the same regions – no explicit future chaining needed.
        global_runtime->execute_index_space(global_ctx, launcher);
    }
};

#endif
