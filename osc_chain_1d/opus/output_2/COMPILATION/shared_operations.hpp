// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <legion.h>

using namespace Legion;

// Common task IDs used across the program
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
};

// Field ID for storing double values in logical regions
enum FieldIDs {
    FID_VAL = 0,
};

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

        // Argument struct for serialization as Legion TaskArgument
        struct TaskArgs {
            Fac1 alpha1;
            Fac2 alpha2;
        };

        // Static Legion task implementation
        // Region requirements:
        //   regions[0]: x1 (WRITE_DISCARD, FID_VAL)
        //   regions[1]: x2 (READ_ONLY,     FID_VAL)
        //   regions[2]: x3 (READ_ONLY,     FID_VAL)
        static void task_body(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
        {
            const TaskArgs *args = reinterpret_cast<const TaskArgs *>(task->args);
            const Fac1 alpha1 = args->alpha1;
            const Fac2 alpha2 = args->alpha2;

            const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1(regions[0], FID_VAL);
            const FieldAccessor<READ_ONLY, double, 1>     acc_x2(regions[1], FID_VAL);
            const FieldAccessor<READ_ONLY, double, 1>     acc_x3(regions[2], FID_VAL);

            Domain dom = runtime->get_index_space_domain(ctx,
                task->regions[0].region.get_index_space());
            Rect<1> rect = dom;

            for (PointInRectIterator<1> pir(rect); pir(); pir++)
            {
                acc_x1[*pir] = alpha1 * acc_x2[*pir] + alpha2 * acc_x3[*pir];
            }
        }
    };
};

#endif
