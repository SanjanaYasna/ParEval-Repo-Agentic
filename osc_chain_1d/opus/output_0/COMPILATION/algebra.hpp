// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// External globals set by the top-level task
extern Runtime *legion_runtime;
extern Context legion_context;

// Common field ID for double values in regions
enum AlgebraFieldIDs {
    ALGEBRA_FID_VAL = 0,
};

// Task ID for the scale_sum2 element-wise operation
enum AlgebraTaskIDs {
    SCALE_SUM2_TASK_ID = 20,
};

// Arguments passed to the scale_sum2 Legion task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool aliased; // true when s1 and s2 refer to the same state
};

// Legion task implementing the scale_sum2 operation on a single block.
// When aliased (s1 == s2): regions[0] = s1/s2 (READ_WRITE), regions[1] = s3 (READ_ONLY)
// When not aliased:        regions[0] = s1 (READ_WRITE), regions[1] = s2 (READ_ONLY),
//                          regions[2] = s3 (READ_ONLY)
// Computes: s1[j] = alpha1 * s2[j] + alpha2 * s3[j]  for each element j in the block.
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    IndexSpace is = task->regions[0].region.get_index_space();
    Domain dom = runtime->get_index_space_domain(ctx, is);
    Rect<1> rect(dom);

    if (args.aliased) {
        // s1 and s2 are the same region: s1 = alpha1*s1 + alpha2*s3
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], ALGEBRA_FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s3(regions[1], ALGEBRA_FID_VAL);

        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s1[*pir] + args.alpha2 * acc_s3[*pir];
        }
    } else {
        // All three regions are distinct: s1 = alpha1*s2 + alpha2*s3
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], ALGEBRA_FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s2(regions[1], ALGEBRA_FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s3(regions[2], ALGEBRA_FID_VAL);

        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
        }
    }
}

struct local_dataflow_algebra
{
    // Applies op element-wise: s1[i] = op(s1[i], s2[i], s3[i]) for each block i.
    // In the symplectic stepper, Op is always scale_sum2 with members m_alpha1, m_alpha2.
    // The state type S is expected to provide:
    //   - size()        : number of blocks
    //   - operator[](i) : LogicalRegion for block i (a subregion)
    //   - region         : parent LogicalRegion (used as the parent in RegionRequirement)
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        const size_t N = s1.size();
        // Detect aliasing: the symplectic stepper often calls for_each3(x, x, y, op)
        // meaning s1 and s2 are the same state (self-update pattern).
        bool aliased = (s1.region == s2.region);

        ScaleSum2Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.aliased = aliased;

        for( size_t i = 0 ; i < N ; ++i )
        {
            TaskLauncher launcher(SCALE_SUM2_TASK_ID,
                                  TaskArgument(&args, sizeof(ScaleSum2Args)));

            // s1[i]: READ_WRITE (output, also read when aliased with s2)
            launcher.add_region_requirement(
                RegionRequirement(s1[i], READ_WRITE, EXCLUSIVE, s1.region));
            launcher.region_requirements[0].add_field(ALGEBRA_FID_VAL);

            if (!aliased) {
                // s2[i]: READ_ONLY (separate input)
                launcher.add_region_requirement(
                    RegionRequirement(s2[i], READ_ONLY, EXCLUSIVE, s2.region));
                launcher.region_requirements[1].add_field(ALGEBRA_FID_VAL);
            }

            // s3[i]: READ_ONLY
            unsigned s3_idx = aliased ? 1 : 2;
            launcher.add_region_requirement(
                RegionRequirement(s3[i], READ_ONLY, EXCLUSIVE, s3.region));
            launcher.region_requirements[s3_idx].add_field(ALGEBRA_FID_VAL);

            // Non-blocking launch; Legion runtime resolves data dependencies
            // automatically based on region access patterns.
            legion_runtime->execute_task(legion_context, launcher);
        }
    }

    // Register the scale_sum2 task with the Legion runtime.
    // Must be called before Runtime::start().
    static void register_tasks()
    {
        {
            TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
            registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
            Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
        }
    }
};

#endif
