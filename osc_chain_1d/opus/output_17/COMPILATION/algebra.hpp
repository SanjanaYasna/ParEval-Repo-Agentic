// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Task ID for the scale_sum2 operation, defined in odeint.cpp
extern const TaskID SCALE_SUM2_TASK_ID;

// Arguments passed to the scale_sum2 index task
struct ScaleSum2TaskArgs {
    double alpha1;
    double alpha2;
    FieldID fid;
    bool in_place; // true when s1 and s2 share the same logical region
};

// Legion task implementing: s1[i] = alpha1 * s2[i] + alpha2 * s3[i]
// When in_place is true, s1 and s2 are the same region (regions[0]),
// and s3 is regions[1].  Otherwise s1=regions[0], s2=regions[1], s3=regions[2].
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2TaskArgs &args =
        *(const ScaleSum2TaskArgs *)task->args;
    const FieldID fid = args.fid;

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], fid);

    if (args.in_place) {
        // s1 == s2: read the "s2" values from the same region as s1
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[1], fid);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            double val = acc_s1[*pir];
            acc_s1[*pir] = args.alpha1 * val + args.alpha2 * acc_s3[*pir];
        }
    } else {
        // All three regions are distinct
        const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], fid);
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], fid);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
        }
    }
}

struct local_dataflow_algebra
{
    // Applies op element-wise across partitioned state:
    //   s1[i] = op(s1[i], s2[i], s3[i])  for every partition block i
    //
    // Op is expected to be scale_sum2 with public members m_alpha1, m_alpha2.
    // S is expected to be state_type (defined in shared_resize.hpp) with
    // members: runtime, ctx, lr, lp, color_space, fid.
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context ctx      = s1.ctx;

        const bool in_place = (s1.lr == s2.lr);

        ScaleSum2TaskArgs args;
        args.alpha1   = op.m_alpha1;
        args.alpha2   = op.m_alpha2;
        args.fid      = s1.fid;
        args.in_place = in_place;

        IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                               s1.color_space,
                               TaskArgument(&args, sizeof(args)),
                               ArgumentMap());

        // Region 0: s1 — read/write destination
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.add_field(0, s1.fid);

        if (!in_place) {
            // Region 1: s2 — read-only source (distinct from s1)
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.add_field(1, s2.fid);
        }

        // Region for s3 — read-only source
        int s3_region_idx = in_place ? 1 : 2;
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0 /*identity projection*/,
                              READ_ONLY, EXCLUSIVE, s3.lr));
        launcher.add_field(s3_region_idx, s3.fid);

        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
