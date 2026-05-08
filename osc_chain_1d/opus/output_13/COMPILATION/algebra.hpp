// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"
#include <cassert>
#include <vector>

using namespace Legion;

// Task and field IDs (defined in odeint.cpp)
extern const TaskID SCALE_SUM2_TASK_ID;
extern const FieldID FID_VAL;

// Arguments passed to the scale_sum2 index task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    bool s1_eq_s2; // true when s1 and s2 refer to the same logical region
};

// scale_sum2 index task implementation:
// Computes x1[i] = alpha1 * x2[i] + alpha2 * x3[i] for each element
// in the partition assigned to this point task.
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    if (args.s1_eq_s2) {
        // s1 and s2 are the same region:
        //   regions[0] = s1 = s2 (READ_WRITE)
        //   regions[1] = s3       (READ_ONLY)
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s3(regions[1], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s1[*pir] + args.alpha2 * acc_s3[*pir];
        }
    } else {
        // All three regions are distinct:
        //   regions[0] = s1 (READ_WRITE)
        //   regions[1] = s2 (READ_ONLY)
        //   regions[2] = s3 (READ_ONLY)
        const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s2(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY,  double, 1> acc_s3(regions[2], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
        }
    }
}

struct local_dataflow_algebra
{
    Runtime *runtime;
    Context ctx;

    local_dataflow_algebra() : runtime(nullptr) {}
    local_dataflow_algebra(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    // for_each3: applies op element-wise across partitioned state vectors.
    // Op is expected to have m_alpha1 and m_alpha2 members (scale_sum2).
    // S is expected to have lr, lp, color_space members (state_type).
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        assert(runtime != nullptr);

        ScaleSum2Args args;
        args.alpha1   = op.m_alpha1;
        args.alpha2   = op.m_alpha2;
        args.s1_eq_s2 = (s1.lr == s2.lr);

        IndexTaskLauncher launcher(SCALE_SUM2_TASK_ID,
                                   s1.color_space,
                                   TaskArgument(&args, sizeof(ScaleSum2Args)),
                                   ArgumentMap());

        // Region 0: s1 (READ_WRITE – output, also serves as s2 input when s1==s2)
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        if (!args.s1_eq_s2) {
            // Region 1: s2 (READ_ONLY – distinct from s1)
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(FID_VAL);

            // Region 2: s3 (READ_ONLY)
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[2].add_field(FID_VAL);
        } else {
            // Region 1: s3 (READ_ONLY – s2 is accessed via region 0)
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[1].add_field(FID_VAL);
        }

        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
