// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"
#include "shared_resize.hpp"

using namespace Legion;

// Arguments for the for_each3 scale_sum2 task
struct ForEach3Args {
    double alpha1;
    double alpha2;
};

struct local_dataflow_algebra
{
    Runtime *runtime;
    Context ctx;

    local_dataflow_algebra() : runtime(nullptr) {}
    local_dataflow_algebra(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    // Applies op element-wise: s1[j] = alpha1*s2[j] + alpha2*s3[j]
    // across all sub-regions of the partition via an index space launch.
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        ForEach3Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;

        IndexLauncher launcher(FOR_EACH3_TASK_ID, s1.color_space,
                               TaskArgument(&args, sizeof(ForEach3Args)),
                               ArgumentMap());

        // s1: output (write-discard, identity projection over partition)
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              WRITE_DISCARD, EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(FID_DATA);

        // s2: input (read-only)
        launcher.add_region_requirement(
            RegionRequirement(s2.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s2.lr));
        launcher.region_requirements[1].add_field(FID_DATA);

        // s3: input (read-only)
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s3.lr));
        launcher.region_requirements[2].add_field(FID_DATA);

        runtime->execute_index_space(ctx, launcher);
    }
};

// Task implementation for for_each3 with scale_sum2 operation:
//   out[i] = alpha1 * in1[i] + alpha2 * in2[i]
// Must be registered with FOR_EACH3_TASK_ID before use.
inline void for_each3_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(ForEach3Args));
    const ForEach3Args *args =
        static_cast<const ForEach3Args *>(task->args);
    const double alpha1 = args->alpha1;
    const double alpha2 = args->alpha2;

    const FieldAccessor<WRITE_DISCARD, double, 1> acc_out(regions[0], FID_DATA);
    const FieldAccessor<READ_ONLY, double, 1>     acc_in1(regions[1], FID_DATA);
    const FieldAccessor<READ_ONLY, double, 1>     acc_in2(regions[2], FID_DATA);

    Rect<1> rect = runtime->get_index_space_domain(
        regions[0].get_logical_region().get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
        acc_out[*pir] = alpha1 * acc_in1[*pir] + alpha2 * acc_in2[*pir];
    }
}

#endif
