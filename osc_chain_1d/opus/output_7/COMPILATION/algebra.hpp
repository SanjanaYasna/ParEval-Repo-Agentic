// Copyright 2013 Mario Mulansky
// Algebra for odeint – Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// ---------------------------------------------------------------------------
// Identifiers shared across the translated application.
// Field ID for the double-valued data stored in every logical region.
// ---------------------------------------------------------------------------
enum FieldIDs {
    FID_X = 0,
};

// Task ID for the scale_sum2 index-space task (implementation in
// shared_operations.hpp, registration in odeint.cpp).
enum AlgebraTaskIDs {
    SCALE_SUM2_TASK_ID = 10,
};

// ---------------------------------------------------------------------------
// Serialisable argument block passed to each point of the scale_sum2 task.
// ---------------------------------------------------------------------------
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// ---------------------------------------------------------------------------
// Algebra plug-in for boost::numeric::odeint.
//
// Replaces every per-block HPX dataflow with a single Legion index-space
// launch whose region requirements let the runtime discover parallelism and
// order the tasks correctly with respect to the rest of the program.
//
// Template parameter S  – state_type (see shared_resize.hpp).
//                         Expected members: runtime, ctx, lr, lp, color_domain
// Template parameter Op – operation   (see shared_operations.hpp).
//                         Expected members: m_alpha1, m_alpha2
// ---------------------------------------------------------------------------
struct local_dataflow_algebra
{
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context  ctx     = s1.ctx;

        // Pack the two scale-sum coefficients into a task argument.
        ScaleSum2Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;

        // One point task per partition colour (i.e. per block of G doubles),
        // mirroring the original per-block HPX dataflow launches.
        IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                               s1.color_domain,
                               TaskArgument(&args, sizeof(ScaleSum2Args)),
                               ArgumentMap());

        // Region requirement 0 – s1 (output, write-discard).
        // The old contents are fully overwritten: x1 = α1·x2 + α2·x3.
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              WRITE_DISCARD, EXCLUSIVE, s1.lr));
        launcher.add_field(0, FID_X);

        // Region requirement 1 – s2 (first input, read-only).
        launcher.add_region_requirement(
            RegionRequirement(s2.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s2.lr));
        launcher.add_field(1, FID_X);

        // Region requirement 2 – s3 (second input, read-only).
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s3.lr));
        launcher.add_field(2, FID_X);

        // Non-blocking: the runtime will order this launch with respect to
        // subsequent launches that touch the same regions, just as the HPX
        // dataflow graph ordered dependent futures.
        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
