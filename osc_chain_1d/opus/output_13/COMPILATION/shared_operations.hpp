// Copyright 2013 Mario Mulansky
// Translated to Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include "legion.h"

using namespace Legion;

typedef std::vector<double> dvec;

// Shared field IDs used across the application
enum FieldIDs {
    FID_VAL = 0,
};

// Shared task IDs used across the application
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    ENERGY_BLOCK_TASK_ID,
};

// Arguments passed to the scale_sum2 Legion task via TaskArgument
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
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

        // Legion task implementation: computes x1[i] = alpha1*x2[i] + alpha2*x3[i]
        // Region requirements (set up by the algebra when launching):
        //   regions[0]: x1 (READ_WRITE), field FID_VAL
        //   regions[1]: x2 (READ_ONLY),  field FID_VAL
        //   regions[2]: x3 (READ_ONLY),  field FID_VAL
        static void cpu_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime)
        {
            const ScaleSum2Args *args =
                reinterpret_cast<const ScaleSum2Args *>(task->args);
            const double a1 = args->alpha1;
            const double a2 = args->alpha2;

            const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
            const FieldAccessor<READ_ONLY,  double, 1> acc_x2(regions[1], FID_VAL);
            const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[2], FID_VAL);

            Domain dom = runtime->get_index_space_domain(ctx,
                task->regions[0].region.get_index_space());

            for (PointInDomainIterator<1> pir(dom); pir(); pir++)
            {
                acc_x1[*pir] = a1 * acc_x2[*pir] + a2 * acc_x3[*pir];
            }
        }
    };
};

#endif
