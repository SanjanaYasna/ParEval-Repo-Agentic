// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cstddef>
#include <cassert>
#include "legion.h"

using namespace Legion;

// Field ID for double values stored in logical regions
enum FieldIDs {
    FID_VAL = 101,
};

// Task IDs for all tasks in the application
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    ENERGY_BLOCK_TASK_ID,
};

// Serializable arguments for the scale_sum2 task
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// Operations struct, parameterizes the odeint stepper
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

        // Apply operation directly using Legion field accessors over a rect
        template< typename Acc1 , typename Acc2 , typename Acc3 >
        void apply( const Acc1 &acc_x1 , const Acc2 &acc_x2 , const Acc3 &acc_x3 ,
                    const Rect<1> &rect ) const
        {
            for( PointInRectIterator<1> pir(rect) ; pir() ; pir++ )
                acc_x1[*pir] = m_alpha1 * acc_x2[*pir] + m_alpha2 * acc_x3[*pir];
        }
    };
};

// Legion task implementation for scale_sum2: x1 = alpha1*x2 + alpha2*x3
// Region requirements:
//   regions[0]: x1 (READ_WRITE)  — destination, overwritten
//   regions[1]: x2 (READ_ONLY)
//   regions[2]: x3 (READ_ONLY)
// Task arguments: ScaleSum2Args { alpha1, alpha2 }
inline void scale_sum2_task( const Task *task ,
                             const std::vector<PhysicalRegion> &regions ,
                             Context ctx , Runtime *runtime )
{
    assert( regions.size() == 3 );
    assert( task->arglen == sizeof(ScaleSum2Args) );
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> acc_x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for( PointInRectIterator<1> pir(rect) ; pir() ; pir++ )
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
}

#endif
