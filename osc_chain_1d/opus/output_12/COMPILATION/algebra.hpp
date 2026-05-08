// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Task ID for the for_each3 (scale_sum2) operation
enum {
    FOR_EACH3_TASK_ID = 2,
};

// Field ID shared across the program for double-valued state data
enum FieldIDs {
    FID_VAL = 0,
};

// Serializable arguments for the for_each3 scale_sum2 task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// Task implementation: s1[i] = alpha1 * s2[i] + alpha2 * s3[i]
inline void for_each3_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<WRITE_DISCARD, double, 1> acc_s1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
    }
}

struct local_dataflow_algebra
{
    // for_each3: applies op (scale_sum2) element-wise across three partitioned states.
    // S is expected to have: runtime (Runtime*), ctx (Context),
    //     lr (LogicalRegion), lp (LogicalPartition), color_space (IndexSpace)
    // Op is expected to have: m_alpha1 (Fac1), m_alpha2 (Fac2)
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        ScaleSum2Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;

        IndexTaskLauncher launcher(FOR_EACH3_TASK_ID, s1.color_space,
                                   TaskArgument(&args, sizeof(ScaleSum2Args)),
                                   ArgumentMap());

        // s1 is output (write-discard), s2 and s3 are read-only inputs
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              WRITE_DISCARD, EXCLUSIVE, s1.lr));
        launcher.add_region_requirement(
            RegionRequirement(s2.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s2.lr));
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s3.lr));

        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.region_requirements[2].add_field(FID_VAL);

        runtime->execute_index_space(ctx, launcher);
    }

    // Pre-register the for_each3 task with the Legion runtime.
    // Must be called before Runtime::start().
    static void register_tasks()
    {
        TaskVariantRegistrar registrar(FOR_EACH3_TASK_ID, "for_each3_scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<for_each3_task>(registrar,
                                                          "for_each3_scale_sum2");
    }
};

#endif
