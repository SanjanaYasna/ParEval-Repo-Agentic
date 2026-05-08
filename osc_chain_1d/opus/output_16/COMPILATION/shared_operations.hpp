// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include "legion.h"
#include <vector>
#include <cstddef>
#include <cassert>

using namespace Legion;

// Common field ID for double values stored in state regions
enum FieldIDs {
    FID_VAL = 0,
};

// Task ID for the scale_sum2 operation
enum OperationTaskIDs {
    SCALE_SUM2_TASK_ID = 10,
};

// Serializable argument struct for scale_sum2 task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

struct local_dataflow_shared_operations
{
    template<typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1(0), m_alpha2(0)
        {}

        scale_sum2(Fac1 alpha1, Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2)
        {}

        // Get the Legion task ID for this operation
        static TaskID task_id() { return SCALE_SUM2_TASK_ID; }

        // Pack arguments for a Legion task launch
        ScaleSum2Args pack_args() const
        {
            return ScaleSum2Args{static_cast<double>(m_alpha1),
                                 static_cast<double>(m_alpha2)};
        }

        // Legion task body
        // Region requirements:
        //   regions[0]: x1 (READ_WRITE, FID_VAL)
        //   regions[1]: x2 (READ_ONLY,  FID_VAL)
        //   regions[2]: x3 (READ_ONLY,  FID_VAL)
        // Computes: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        static void cpu_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
        {
            assert(regions.size() == 3);
            assert(task->arglen == sizeof(ScaleSum2Args));

            const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

            const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
            const FieldAccessor<READ_ONLY, double, 1>  acc_x2(regions[1], FID_VAL);
            const FieldAccessor<READ_ONLY, double, 1>  acc_x3(regions[2], FID_VAL);

            Domain dom = runtime->get_index_space_domain(ctx,
                task->regions[0].region.get_index_space());
            Rect<1> rect = dom;

            for (PointInRectIterator<1> pir(rect); pir(); pir++)
            {
                acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
            }
        }

        // Register this task with the Legion runtime (call before Runtime::start)
        static void register_task()
        {
            TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
            registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
            Runtime::preregister_task_variant<cpu_task>(registrar, "scale_sum2");
        }
    };
};

#endif
