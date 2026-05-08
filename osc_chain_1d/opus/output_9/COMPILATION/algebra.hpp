// Copyright 2013 Mario Mulansky
#ifndef LEGION_ALGEBRA_HPP
#define LEGION_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Field ID for double values in regions – must be consistent across all files
enum {
    FID_VAL = 0,
};

// Task ID for the scale_sum2 operation
enum {
    SCALE_SUM2_TASK_ID = 100,
};

// Arguments passed to the scale_sum2 task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool in_place; // true when s1 and s2 are the same state
};

// Legion task implementing the scale_sum2 operation on a single block.
// When in_place (s1 == s2): x1 = alpha1 * x1 + alpha2 * x3  (2 region reqs)
// Otherwise:                x1 = alpha1 * x2 + alpha2 * x3  (3 region reqs)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args *args = (const ScaleSum2Args *)task->args;

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    if (args->in_place) {
        // s1 == s2 case: read-modify-write on region 0, read from region 1
        const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1>  acc_x3(regions[1], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_x1[*pir] = args->alpha1 * acc_x1[*pir]
                         + args->alpha2 * acc_x3[*pir];
        }
    } else {
        // General case: write region 0, read regions 1 and 2
        const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1(regions[0], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1>     acc_x2(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1>     acc_x3(regions[2], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_x1[*pir] = args->alpha1 * acc_x2[*pir]
                         + args->alpha2 * acc_x3[*pir];
        }
    }
}

// Legion algebra: replaces HPX local_dataflow_algebra.
// Launches Legion tasks to perform element-wise operations on partitioned states.
// The Legion runtime infers data-flow dependencies from region requirements,
// mirroring the future-chain semantics of the HPX dataflow algebra.
struct legion_algebra
{
    // Applies op element-wise across all blocks of s1, s2, s3.
    // Op is expected to have m_alpha1 and m_alpha2 members (e.g. scale_sum2).
    // S is the state_type, assumed to expose:
    //   size()      – number of blocks
    //   blocks[i]   – LogicalRegion for block i (sub-region of a partition)
    //   region      – top-level LogicalRegion (parent for privileges)
    //   runtime     – Legion Runtime*
    //   ctx         – Legion Context
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        const size_t N = s1.size();
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        // Detect aliasing: the symplectic stepper commonly passes the same
        // state as both s1 and s2  (e.g. q = 1*q + dt*a*p).
        const bool in_place = (&s1 == &s2);

        for (size_t i = 0; i < N; ++i)
        {
            ScaleSum2Args args;
            args.alpha1  = op.m_alpha1;
            args.alpha2  = op.m_alpha2;
            args.in_place = in_place;

            TaskLauncher launcher(SCALE_SUM2_TASK_ID,
                TaskArgument(&args, sizeof(ScaleSum2Args)));

            if (in_place) {
                // s1 == s2 → two region requirements
                // Region 0: s1/s2 block – READ_WRITE (in-place update)
                launcher.add_region_requirement(
                    RegionRequirement(s1.blocks[i], READ_WRITE,
                                      EXCLUSIVE, s1.region));
                launcher.region_requirements[0].add_field(FID_VAL);

                // Region 1: s3 block – READ_ONLY
                launcher.add_region_requirement(
                    RegionRequirement(s3.blocks[i], READ_ONLY,
                                      EXCLUSIVE, s3.region));
                launcher.region_requirements[1].add_field(FID_VAL);
            } else {
                // All distinct → three region requirements
                // Region 0: s1 block – WRITE_DISCARD
                launcher.add_region_requirement(
                    RegionRequirement(s1.blocks[i], WRITE_DISCARD,
                                      EXCLUSIVE, s1.region));
                launcher.region_requirements[0].add_field(FID_VAL);

                // Region 1: s2 block – READ_ONLY
                launcher.add_region_requirement(
                    RegionRequirement(s2.blocks[i], READ_ONLY,
                                      EXCLUSIVE, s2.region));
                launcher.region_requirements[1].add_field(FID_VAL);

                // Region 2: s3 block – READ_ONLY
                launcher.add_region_requirement(
                    RegionRequirement(s3.blocks[i], READ_ONLY,
                                      EXCLUSIVE, s3.region));
                launcher.region_requirements[2].add_field(FID_VAL);
            }

            // Task is launched asynchronously; the Legion runtime ensures
            // correct ordering via region-based dependence analysis.
            runtime->execute_task(ctx, launcher);
        }
    }
};

#endif
