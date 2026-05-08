// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Shared field ID for double-valued state data
enum StateFieldIDs {
    FID_VAL = 0,
};

// Task IDs used by the algebra (to be registered in main)
enum AlgebraTaskIDs {
    FOR_EACH3_TASK_ID = 10,
};

// Serializable arguments for the for_each3 task (scale_sum2 parameters)
struct ForEach3Args {
    double alpha1;
    double alpha2;
    bool aliased; // true when s1 and s2 refer to the same state
};

// Legion task implementing the scale_sum2 operation on one block.
// If aliased:  regions[0] = s1 (=s2) READ_WRITE, regions[1] = s3 READ_ONLY
// Otherwise:   regions[0] = s1 READ_WRITE, regions[1] = s2 READ_ONLY,
//              regions[2] = s3 READ_ONLY
inline void for_each3_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    const ForEach3Args &args = *(const ForEach3Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> acc_s1(regions[0], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    if (args.aliased) {
        // s1 and s2 are the same region; s3 is regions[1]
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[1], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            double val = acc_s1[*pir];
            acc_s1[*pir] = args.alpha1 * val + args.alpha2 * acc_s3[*pir];
        }
    } else {
        // All three are distinct regions
        const FieldAccessor<READ_ONLY, double, 1> acc_s2(regions[1], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> acc_s3(regions[2], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++) {
            acc_s1[*pir] = args.alpha1 * acc_s2[*pir] + args.alpha2 * acc_s3[*pir];
        }
    }
}

struct local_dataflow_algebra
{
    Runtime *runtime;
    Context ctx;

    local_dataflow_algebra()
        : runtime(nullptr), ctx()
    {}

    local_dataflow_algebra(Runtime *rt, Context c)
        : runtime(rt), ctx(c)
    {}

    // for_each3: applies Op (scale_sum2) element-wise across blocks of s1, s2, s3.
    // Launches one Legion task per block; the runtime resolves data dependencies
    // through region requirements, analogous to HPX dataflow scheduling.
    template< typename S , typename Op >
    void for_each3( S &s1 , const S &s2 , const S &s3 , Op op )
    {
        const size_t N = s1.size();

        ForEach3Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        // Detect aliasing: odeint's symplectic stepper may pass the same
        // state as both s1 and s2 for in-place updates (e.g. q = 1*q + dt*c*p).
        args.aliased = (static_cast<const void*>(&s1) ==
                        static_cast<const void*>(&s2));

        for ( size_t i = 0 ; i < N ; ++i )
        {
            TaskLauncher launcher(FOR_EACH3_TASK_ID,
                                  TaskArgument(&args, sizeof(ForEach3Args)));

            // Region 0: s1[i] — destination (READ_WRITE)
            launcher.add_region_requirement(
                RegionRequirement(s1[i], READ_WRITE, EXCLUSIVE, s1[i]));
            launcher.add_field(0, FID_VAL);

            if ( !args.aliased )
            {
                // Region 1: s2[i] — source (READ_ONLY), distinct from s1
                launcher.add_region_requirement(
                    RegionRequirement(s2[i], READ_ONLY, EXCLUSIVE, s2[i]));
                launcher.add_field(1, FID_VAL);
            }

            // s3[i] — source (READ_ONLY)
            int s3_idx = args.aliased ? 1 : 2;
            launcher.add_region_requirement(
                RegionRequirement(s3[i], READ_ONLY, EXCLUSIVE, s3[i]));
            launcher.add_field(s3_idx, FID_VAL);

            runtime->execute_task(ctx, launcher);
        }
    }
};

#endif
