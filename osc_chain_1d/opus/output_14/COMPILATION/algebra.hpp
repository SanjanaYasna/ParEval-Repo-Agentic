// Copyright 2013 Mario Mulansky
// algebra for odeint – Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"
#include <vector>
#include <cassert>

using namespace Legion;

// Common field ID (must be consistent across all translation units)
#ifndef FID_VAL_DEFINED
#define FID_VAL_DEFINED
enum FieldIDs : FieldID {
    FID_VAL = 0,
};
#endif

enum AlgebraTaskIDs : TaskID {
    SCALE_SUM2_TASK_ID = 10,
};

// Arguments serialized into the task launcher for scale_sum2
struct ScaleSumArgs {
    double alpha1;
    double alpha2;
    bool src1_aliased; // true when s1 and s2 refer to the same logical region
};

// Legion task: dst[i] = alpha1 * src1[i] + alpha2 * src2[i]
// When src1_aliased, src1 shares the dst region (regions[0]).
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(ScaleSumArgs));
    const ScaleSumArgs &args =
        *reinterpret_cast<const ScaleSumArgs *>(task->args);

    const FieldAccessor<READ_WRITE, double, 1> acc_dst(regions[0], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    if (args.src1_aliased) {
        // s1 == s2: read src1 from the dst region, src2 from regions[1]
        const FieldAccessor<READ_ONLY, double, 1> acc_src2(regions[1], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_dst[*pir] =
                args.alpha1 * acc_dst[*pir] + args.alpha2 * acc_src2[*pir];
        }
    } else {
        // All distinct: src1 = regions[1], src2 = regions[2]
        const FieldAccessor<READ_ONLY, double, 1> acc_src1(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> acc_src2(regions[2], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_dst[*pir] =
                args.alpha1 * acc_src1[*pir] + args.alpha2 * acc_src2[*pir];
        }
    }
}

// Must be called before Runtime::start() to register algebra tasks.
inline void preregister_algebra_tasks()
{
    TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
}

// ---------------------------------------------------------------------------
// Algebra used by the symplectic stepper.
//
// In HPX the algebra wrapped every per-block operation in a dataflow that
// produced a new shared_future.  In Legion the same parallelism is expressed
// with an IndexLauncher over the partition colour space; the runtime
// automatically resolves data dependencies through region requirements.
//
// state_type (defined in shared_resize.hpp) is expected to expose:
//   LogicalRegion   lr        – parent logical region
//   LogicalPartition lp       – equal partition into M blocks
//   IndexSpace      color_is  – colour index space (M points)
// ---------------------------------------------------------------------------
struct local_dataflow_algebra
{
    Runtime *runtime;
    Context  ctx;

    local_dataflow_algebra() : runtime(nullptr) {}
    local_dataflow_algebra(Runtime *rt, Context c) : runtime(rt), ctx(c) {}

    // for_each3  –  element-wise:  s1[i] = op( s1[i], s2[i], s3[i] )
    //
    // S  is state_type.
    // Op is local_dataflow_shared_operations::scale_sum2<Fac1,Fac2>,
    //    which carries m_alpha1 and m_alpha2.
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        assert(runtime != nullptr &&
               "local_dataflow_algebra must be initialised with Runtime/Context");

        ScaleSumArgs args;
        args.alpha1       = op.m_alpha1;
        args.alpha2       = op.m_alpha2;
        args.src1_aliased = (s1.lr == s2.lr);

        IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                               s1.color_is,
                               TaskArgument(&args, sizeof(ScaleSumArgs)),
                               ArgumentMap());

        // Region 0 – destination s1 (READ_WRITE)
        launcher.add_region_requirement(
            RegionRequirement(s1.lp, 0 /*identity projection*/,
                              READ_WRITE, EXCLUSIVE, s1.lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        if (!args.src1_aliased) {
            // Region 1 – source s2 (READ_ONLY), only when s2 ≠ s1
            launcher.add_region_requirement(
                RegionRequirement(s2.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s2.lr));
            launcher.region_requirements[1].add_field(FID_VAL);
        }

        // Next region – source s3 (READ_ONLY)
        {
            int s3_idx = args.src1_aliased ? 1 : 2;
            launcher.add_region_requirement(
                RegionRequirement(s3.lp, 0 /*identity projection*/,
                                  READ_ONLY, EXCLUSIVE, s3.lr));
            launcher.region_requirements[s3_idx].add_field(FID_VAL);
        }

        // Launch without blocking – Legion resolves ordering through
        // region requirements (analogous to HPX future chaining).
        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
