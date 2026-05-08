// Copyright 2013 Mario Mulansky
// Operations functionality for odeint – Legion version
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include "legion.h"

using namespace Legion;

// Common field and task IDs shared across the application
enum FieldIDs {
    FID_VAL = 0,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    ENERGY_TASK_ID,
};

// Global runtime and context, set once in the top-level task
extern Runtime *legion_runtime;
extern Context legion_context;

// Serializable arguments for the scale_sum2 Legion task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
    int x2_region_idx;  // index into regions[] for x2 data
    int x3_region_idx;  // index into regions[] for x3 data
};

// Legion task implementation for scale_sum2: x1[i] = alpha1*x2[i] + alpha2*x3[i]
// Handles aliasing (e.g. x1 == x2) via region index mapping in ScaleSum2Args.
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    if (args.x2_region_idx == 0 && args.x3_region_idx == 0) {
        // x1 == x2 == x3
        for (PointInRectIterator<1> pir(rect); pir(); pir++)
            acc_x1[*pir] = args.alpha1 * acc_x1[*pir] + args.alpha2 * acc_x1[*pir];
    } else if (args.x2_region_idx == 0) {
        // x1 == x2, x3 is separate
        const FieldAccessor<READ_ONLY, double, 1> acc_x3(regions[args.x3_region_idx], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++)
            acc_x1[*pir] = args.alpha1 * acc_x1[*pir] + args.alpha2 * acc_x3[*pir];
    } else if (args.x3_region_idx == 0) {
        // x1 == x3, x2 is separate
        const FieldAccessor<READ_ONLY, double, 1> acc_x2(regions[args.x2_region_idx], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++)
            acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x1[*pir];
    } else {
        // All three are distinct regions
        const FieldAccessor<READ_ONLY, double, 1> acc_x2(regions[args.x2_region_idx], FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> acc_x3(regions[args.x3_region_idx], FID_VAL);
        for (PointInRectIterator<1> pir(rect); pir(); pir++)
            acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

struct local_dataflow_shared_operations
{
    template< typename Fac1 , typename Fac2=Fac1 >
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1( 0 ) , m_alpha2( 0 )
        {}

        scale_sum2( Fac1 alpha1 , Fac2 alpha2 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 )
        { }

        // Called by algebra for_each3 on each block.
        // Launches a Legion task that performs the element-wise operation
        // on the LogicalRegion data.
        template< typename S1 , typename S2 , typename S3 >
        S1 operator() ( S1 x1 , const S2 x2 , const S3 x3 ) const
        {
            ScaleSum2Args args;
            args.alpha1 = static_cast<double>(m_alpha1);
            args.alpha2 = static_cast<double>(m_alpha2);

            // Determine region requirement layout, avoiding duplicate
            // requirements for aliased regions (e.g. x1 == x2).
            int next_idx = 1;
            args.x2_region_idx = (x2 == x1) ? 0 : next_idx++;
            if (x3 == x1)
                args.x3_region_idx = 0;
            else if (x3 == x2)
                args.x3_region_idx = args.x2_region_idx;
            else
                args.x3_region_idx = next_idx++;

            TaskLauncher launcher(SCALE_SUM2_TASK_ID,
                                  TaskArgument(&args, sizeof(args)));

            // x1 is always READ_WRITE (written, and possibly also read when aliased)
            launcher.add_region_requirement(
                RegionRequirement(x1, READ_WRITE, EXCLUSIVE, x1));
            launcher.region_requirements[0].add_field(FID_VAL);

            // Add x2 as a separate READ_ONLY requirement if distinct from x1
            if (args.x2_region_idx != 0) {
                launcher.add_region_requirement(
                    RegionRequirement(x2, READ_ONLY, EXCLUSIVE, x2));
                launcher.region_requirements[args.x2_region_idx].add_field(FID_VAL);
            }

            // Add x3 as a separate READ_ONLY requirement if distinct from x1 and x2
            if (args.x3_region_idx != 0 &&
                args.x3_region_idx != args.x2_region_idx) {
                launcher.add_region_requirement(
                    RegionRequirement(x3, READ_ONLY, EXCLUSIVE, x3));
                launcher.region_requirements[args.x3_region_idx].add_field(FID_VAL);
            }

            legion_runtime->execute_task(legion_context, launcher);
            return x1;
        }
    };
};

#endif
