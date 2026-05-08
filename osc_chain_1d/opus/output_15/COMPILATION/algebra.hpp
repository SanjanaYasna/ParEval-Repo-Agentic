// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow algebra to Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <legion.h>

using namespace Legion;

// Task ID for the scale_sum2 index task (registered in main)
enum AlgebraTaskID {
    SCALE_SUM2_TASK_ID = 200,
};

// Serializable arguments for the scale_sum2 task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool self_alias; // true when s1 and s2 refer to the same logical region
    FieldID fid_s1;
    FieldID fid_s2;
    FieldID fid_s3;
};

// Legion index task implementing: s1[i] = alpha1 * s2[i] + alpha2 * s3[i]
// Handles both the aliased case (s1 == s2) and the non-aliased case.
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    if (args.self_alias) {
        // s1 and s2 are the same region
        // Region 0: s1/s2 (READ_WRITE), Region 1: s3 (READ_ONLY)
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], args.fid_s1);
        const FieldAccessor<READ_ONLY, double, 1>  acc_s3(regions[1], args.fid_s3);

        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s1[*pir] + args.alpha2 * acc_s3[*pir];
        }
    } else {
        // All three regions are distinct
        // Region 0: s1 (WRITE_DISCARD), Region 1: s2 (READ_ONLY), Region 2: s3 (READ_ONLY)
        const FieldAccessor<WRITE_DISCARD, double, 1> acc_s1(regions[0], args.fid_s1);
        const FieldAccessor<READ_ONLY, double, 1>     acc_s2(regions[1], args.fid_s2);
        const FieldAccessor<READ_ONLY, double, 1>     acc_s3(regions[2], args.fid_s3);

        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
        }
    }
}

struct local_dataflow_algebra
{
    // Applies op element-wise across three states partitioned into blocks.
    // Op is expected to be scale_sum2 with members m_alpha1 and m_alpha2.
    // In odeint's symplectic stepper, s1 may alias s2 (e.g., q_out == q_in).
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        ScaleSum2Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.fid_s1 = s1.fid;
        args.fid_s2 = s2.fid;
        args.fid_s3 = s3.fid;

        bool self_alias = (s1.lr == s2.lr);
        args.self_alias = self_alias;

        IndexLauncher launcher(SCALE_SUM2_TASK_ID, s1.color_space,
                               TaskArgument(&args, sizeof(args)), ArgumentMap());

        if (self_alias) {
            // s1 and s2 refer to the same logical region — use READ_WRITE
            launcher.add_region_requirement(
                RegionRequirement(s1.lp, 0 /*identity projection*/,
                                  READ_WRITE, EXCLUSIVE, s1.lr));
            launcher.region_requirements[0].add_field(s1.fid);

            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[1].add_field(s3.fid);
        } else {
            // All three states are backed by distinct logical regions
            launcher.add_region_requirement(
                RegionRequirement(s1.lp, 0 /*identity projection*/,
                                  WRITE_DISCARD, EXCLUSIVE, s1.lr));
            launcher.region_requirements[0].add_field(s1.fid);

            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(s2.fid);

            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[2].add_field(s3.fid);
        }

        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
