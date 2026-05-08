// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

enum {
    FOR_EACH3_TASK_ID = 20,
};

struct ForEach3Args {
    double alpha1;
    double alpha2;
    FieldID fid;
    bool s1_is_s2;
};

struct local_dataflow_algebra
{
    // Applies op element-wise: s1[i] = op(s1[i], s2[i], s3[i]) for each partition i.
    // Op is expected to be scale_sum2 with members m_alpha1, m_alpha2.
    // S is expected to be state_type with members: lr, lp, color_space, fid, runtime, ctx.
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        ForEach3Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.fid = s1.fid;
        args.s1_is_s2 = (s1.lr == s2.lr);

        IndexLauncher launcher(FOR_EACH3_TASK_ID, s1.color_space,
                               TaskArgument(&args, sizeof(ForEach3Args)),
                               ArgumentMap());

        // Region 0: s1 (READ_WRITE) — also serves as s2 when s1 == s2
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*projection ID*/, READ_WRITE,
                              EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(s1.fid);

        if (!args.s1_is_s2) {
            // Region 1: s2 (READ_ONLY), only when s1 and s2 are distinct regions
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0, READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(s2.fid);
        }

        // s3: READ_ONLY (index depends on whether s2 was added)
        int s3_idx = args.s1_is_s2 ? 1 : 2;
        launcher.add_region_requirement(
            RegionRequirement(s3.lp, 0, READ_ONLY, EXCLUSIVE, s3.lr));
        launcher.region_requirements[s3_idx].add_field(s3.fid);

        // Launch without explicit wait; Legion runtime enforces correct
        // ordering via region-based dependence analysis.
        runtime->execute_index_space(ctx, launcher);
    }
};

// Index task body: performs s1 = alpha1*s2 + alpha2*s3 on one partition.
inline void for_each3_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    ForEach3Args args = *(const ForEach3Args *)task->args;

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], args.fid);

    if (args.s1_is_s2) {
        // s1 and s2 share the same region (regions[0]),
        // s3 is regions[1]
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[1], args.fid);
        for (PointInRectIterator<1> itr(rect); itr(); itr++) {
            double val = acc_s1[*itr];
            acc_s1[*itr] = args.alpha1 * val + args.alpha2 * acc_s3[*itr];
        }
    } else {
        // All three regions are distinct
        const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], args.fid);
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], args.fid);
        for (PointInRectIterator<1> itr(rect); itr(); itr++) {
            acc_s1[*itr] = args.alpha1 * acc_s2[*itr] + args.alpha2 * acc_s3[*itr];
        }
    }
}

inline void preregister_algebra_tasks()
{
    TaskVariantRegistrar registrar(FOR_EACH3_TASK_ID, "for_each3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<for_each3_task>(registrar, "for_each3");
}

#endif
