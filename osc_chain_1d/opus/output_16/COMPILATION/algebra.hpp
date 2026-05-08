// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Field ID for double data values stored in regions
enum FieldIDs {
    FID_VAL = 0,
};

// Task ID for the scale_sum2 index launch
enum AlgebraTaskIDs {
    SCALE_SUM2_TASK_ID = 2,
};

// Arguments passed to the scale_sum2 point task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool s1_is_s2; // true when s1 and s2 refer to the same logical region
};

// Task implementation for the scale_sum2 operation:
//   s1[i] = alpha1 * s2[i] + alpha2 * s3[i]
//
// When s1_is_s2 is true (in-place update):
//   regions[0] = s1/s2 (READ_WRITE), regions[1] = s3 (READ_ONLY)
// When s1_is_s2 is false (all distinct):
//   regions[0] = s1 (READ_WRITE), regions[1] = s2 (READ_ONLY), regions[2] = s3 (READ_ONLY)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_VAL);

    if (args.s1_is_s2) {
        // s1 and s2 alias the same region; read s2 from regions[0], s3 from regions[1]
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[1], FID_VAL);
        for (PointInRectIterator<1> it(rect); it(); it++) {
            double v1 = acc_s1[*it];
            double v3 = acc_s3[*it];
            acc_s1[*it] = args.alpha1 * v1 + args.alpha2 * v3;
        }
    } else {
        // All three states are backed by distinct regions
        const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], FID_VAL);
        for (PointInRectIterator<1> it(rect); it(); it++) {
            double v2 = acc_s2[*it];
            double v3 = acc_s3[*it];
            acc_s1[*it] = args.alpha1 * v2 + args.alpha2 * v3;
        }
    }
}

struct local_dataflow_algebra
{
    Runtime *runtime;
    Context ctx;

    local_dataflow_algebra() : runtime(nullptr) {}
    local_dataflow_algebra(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    // for_each3: applies op(s1[i], s2[i], s3[i]) for each partition i
    // S is expected to be state_type with members: region, partition, color_space
    // Op is expected to have public members: m_alpha1, m_alpha2
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        const bool s1_is_s2 = (s1.region == s2.region);

        ScaleSum2Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.s1_is_s2 = s1_is_s2;

        IndexLauncher launcher(SCALE_SUM2_TASK_ID, s1.color_space,
                               TaskArgument(&args, sizeof(args)), ArgumentMap());

        // Region requirement 0: s1 (READ_WRITE); also serves as s2 when aliased
        {
            RegionRequirement req(s1.partition, 0 /*identity projection*/,
                                  READ_WRITE, EXCLUSIVE, s1.region);
            req.add_field(FID_VAL);
            launcher.add_region_requirement(req);
        }

        // If s1 and s2 are distinct, add s2 as a separate READ_ONLY requirement
        if (!s1_is_s2) {
            RegionRequirement req(s2.partition, 0, READ_ONLY, EXCLUSIVE, s2.region);
            req.add_field(FID_VAL);
            launcher.add_region_requirement(req);
        }

        // Region requirement for s3 (READ_ONLY)
        {
            RegionRequirement req(s3.partition, 0, READ_ONLY, EXCLUSIVE, s3.region);
            req.add_field(FID_VAL);
            launcher.add_region_requirement(req);
        }

        runtime->execute_index_space(ctx, launcher);
    }

    // Register algebra tasks; call before Runtime::start
    static void preregister_tasks()
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
};

#endif
