// Copyright 2013 Mario Mulansky
// Algebra for odeint – Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Field ID for the double data stored in regions
enum FieldIDs {
    FID_DATA = 101,
};

// Task ID for the scale_sum2 algebraic operation
enum AlgebraTaskIDs {
    SCALE_SUM2_TASK_ID = 201,
};

// Serializable arguments for the scale_sum2 task
struct ScaleSumArgs {
    double alpha1;
    double alpha2;
    bool self_update; // true when s1 and s2 refer to the same logical region
};

// Legion task implementing the scale_sum2 operation on a single partition block.
// When self_update is true:
//   regions[0] = s1 (READ_WRITE, also serves as s2), regions[1] = s3 (READ_ONLY)
// When self_update is false:
//   regions[0] = s1 (READ_WRITE), regions[1] = s2 (READ_ONLY), regions[2] = s3 (READ_ONLY)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSumArgs &args = *(const ScaleSumArgs *)task->args;

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    if (args.self_update) {
        // s1 == s2: in-place update  s1[j] = alpha1 * s1[j] + alpha2 * s3[j]
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_DATA);
        const FieldAccessor<READ_ONLY, double, 1>  acc_s3(regions[1], FID_DATA);

        for (PointInRectIterator<1> it(rect); it(); it++) {
            acc_s1[*it] = args.alpha1 * acc_s1[*it] + args.alpha2 * acc_s3[*it];
        }
    } else {
        // All distinct: s1[j] = alpha1 * s2[j] + alpha2 * s3[j]
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_DATA);
        const FieldAccessor<READ_ONLY, double, 1>  acc_s2(regions[1], FID_DATA);
        const FieldAccessor<READ_ONLY, double, 1>  acc_s3(regions[2], FID_DATA);

        for (PointInRectIterator<1> it(rect); it(); it++) {
            acc_s1[*it] = args.alpha1 * acc_s2[*it] + args.alpha2 * acc_s3[*it];
        }
    }
}

// Algebra struct for use with boost::numeric::odeint.
// Replaces HPX dataflow-based algebra with Legion index task launches.
// Assumes state_type S exposes:
//   Runtime *runtime, Context ctx,
//   LogicalRegion logical_region, LogicalPartition logical_partition,
//   IndexSpace color_space, size_t num_partitions
struct local_dataflow_algebra
{
    // Op is expected to be local_dataflow_shared_operations::scale_sum2<Fac1,Fac2>
    // with public members m_alpha1 and m_alpha2.
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        // Detect in-place update (odeint symplectic stepper commonly passes s1 == s2)
        bool self_update = (s1.logical_region == s2.logical_region);

        ScaleSumArgs args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.self_update = self_update;

        IndexLauncher launcher(SCALE_SUM2_TASK_ID, s1.color_space,
                               TaskArgument(&args, sizeof(ScaleSumArgs)),
                               ArgumentMap());

        // Region 0: s1 destination (READ_WRITE)
        launcher.add_region_requirement(
            RegionRequirement(s1.logical_partition, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.logical_region));
        launcher.region_requirements[0].add_field(FID_DATA);

        if (self_update) {
            // s2 is the same region as s1, so only add s3 as region 1
            launcher.add_region_requirement(
                RegionRequirement(s3.logical_partition, 0,
                                  READ_ONLY, EXCLUSIVE, s3.logical_region));
            launcher.region_requirements[1].add_field(FID_DATA);
        } else {
            // Region 1: s2 source (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s2.logical_partition, 0,
                                  READ_ONLY, EXCLUSIVE, s2.logical_region));
            launcher.region_requirements[1].add_field(FID_DATA);

            // Region 2: s3 source (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s3.logical_partition, 0,
                                  READ_ONLY, EXCLUSIVE, s3.logical_region));
            launcher.region_requirements[2].add_field(FID_DATA);
        }

        // Launch asynchronously – Legion runtime manages ordering via
        // region-based dependency tracking (replacing HPX future chaining)
        runtime->execute_index_space(ctx, launcher);
    }

    // Must be called before Runtime::start() to pre-register algebra tasks
    static void register_tasks()
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
};

#endif
