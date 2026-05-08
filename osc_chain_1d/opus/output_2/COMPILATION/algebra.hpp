// Copyright 2013 Mario Mulansky
// Algebra for odeint — Legion version
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"
#include <cstring>
#include <cassert>

using namespace Legion;

// Common task and field IDs shared across translation units.
// Guarded so they can appear in multiple headers safely.
#ifndef LEGION_ODEINT_IDS
#define LEGION_ODEINT_IDS
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
};
enum FieldIDs {
    FID_VAL = 0,
};
#endif

// Arguments passed to the scale_sum2 index task.
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool s1_eq_s2; // true when s1 and s2 alias the same logical region
};

// scale_sum2 index task implementation.
//
// When s1_eq_s2 == true (s1 and s2 are the same region):
//   regions[0] = s1/s2  (READ_WRITE)
//   regions[1] = s3     (READ_ONLY)
//   Computes: s1[i] = alpha1 * s1[i] + alpha2 * s3[i]
//
// When s1_eq_s2 == false (all regions distinct):
//   regions[0] = s1  (READ_WRITE)
//   regions[1] = s2  (READ_ONLY)
//   regions[2] = s3  (READ_ONLY)
//   Computes: s1[i] = alpha1 * s2[i] + alpha2 * s3[i]
//
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(ScaleSum2Args));
    ScaleSum2Args args;
    memcpy(&args, task->args, sizeof(ScaleSum2Args));

    const Rect<1> rect =
        runtime->get_index_space_domain(ctx,
            task->regions[0].region.get_index_space());

    const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_VAL);

    if (args.s1_eq_s2) {
        // s1 and s2 share the same region (common in symplectic integrators
        // where q_out == q_in or p_out == p_in).
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[1], FID_VAL);
        for (PointInRectIterator<1> itr(rect); itr(); itr++) {
            const double v = acc_s1[*itr];
            acc_s1[*itr] = args.alpha1 * v + args.alpha2 * acc_s3[*itr];
        }
    } else {
        // All three regions are distinct.
        const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], FID_VAL);
        for (PointInRectIterator<1> itr(rect); itr(); itr++) {
            acc_s1[*itr] = args.alpha1 * acc_s2[*itr] + args.alpha2 * acc_s3[*itr];
        }
    }
}

// Algebra struct used by boost::numeric::odeint steppers.
//
// Assumes the state type S has the following members:
//   - Runtime *runtime
//   - Context  ctx
//   - LogicalRegion   lr        (parent logical region)
//   - LogicalPartition lp       (partition into M blocks)
//   - IndexSpace       color_is (M-point color space of the partition)
//
// The operation Op is expected to expose m_alpha1 and m_alpha2 members
// (satisfied by local_dataflow_shared_operations::scale_sum2).
struct local_dataflow_algebra
{
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        // Detect aliasing: in the symplectic stepper the pattern
        //   for_each3(q, q, dqdt, ...)  (s1 == s2)
        // is common.  We must avoid placing the same logical partition
        // into the launcher twice with conflicting privileges.
        const bool eq12 = (s1.lr == s2.lr);

        // s1 == s3 aliasing does not arise in the symplectic stepper;
        // guard against it just in case.
        assert(!(s1.lr == s3.lr) || eq12);

        ScaleSum2Args args;
        args.alpha1   = op.m_alpha1;
        args.alpha2   = op.m_alpha2;
        args.s1_eq_s2 = eq12;

        // Build the index launcher over the block color space.
        // Projection functor 0 (identity) maps each color point to
        // the corresponding sub-region of the partition.
        IndexLauncher launcher(SCALE_SUM2_TASK_ID, s1.color_is,
                               TaskArgument(&args, sizeof(ScaleSum2Args)),
                               ArgumentMap());

        // Region requirement 0: s1 — always READ_WRITE.
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*projection ID*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        if (!eq12) {
            // Region requirement 1: s2 — READ_ONLY (only when distinct
            // from s1; otherwise s2 data is accessed through req 0).
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*projection ID*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(FID_VAL);
        }

        // Next region requirement: s3 — READ_ONLY.
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0 /*projection ID*/,
                              READ_ONLY, EXCLUSIVE, s3.lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Launch — non-blocking.  Legion will enforce data dependence
        // ordering with any subsequent tasks that touch the same
        // (sub-)regions, mirroring the implicit dependence graph that
        // HPX futures created via dataflow.
        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
