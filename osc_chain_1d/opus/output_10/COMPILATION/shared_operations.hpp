// Copyright 2013 Mario Mulansky
// operations functionality for odeint – Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cassert>
#include <legion.h>

using namespace Legion;

// Common field ID for double values stored in regions
enum FieldIDs {
    FID_VAL = 0,
};

// Task IDs used throughout the codebase
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

struct local_dataflow_shared_operations
{
    template< typename Fac1 , typename Fac2=Fac1 >
    struct scale_sum2
    {
        Fac1 m_alpha1;
        Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1( 0 ) , m_alpha2( 0 )
        {}

        scale_sum2( Fac1 alpha1 , Fac2 alpha2 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 )
        { }

        // Legion task implementation for scale_sum2.
        // Supports two modes based on the number of physical regions:
        //
        //   2 regions (in-place, used when x1 == x2):
        //     Region 0: x1 (READ_WRITE), Region 1: x3 (READ_ONLY)
        //     Computes x1[i] = alpha1 * x1[i] + alpha2 * x3[i]
        //
        //   3 regions (out-of-place, x1 != x2):
        //     Region 0: x1 (WRITE_DISCARD), Region 1: x2 (READ_ONLY), Region 2: x3 (READ_ONLY)
        //     Computes x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        static void cpu_task( const Task *task ,
                              const std::vector<PhysicalRegion> &regions ,
                              Context ctx , Runtime *runtime )
        {
            assert(task->arglen == sizeof(scale_sum2));
            const scale_sum2 &op =
                *reinterpret_cast<const scale_sum2*>(task->args);

            Domain dom = runtime->get_index_space_domain( ctx ,
                task->regions[0].region.get_index_space() );
            Rect<1> rect = dom;

            if ( regions.size() == 2 )
            {
                // In-place mode: x1 and x2 are the same region
                const FieldAccessor<READ_WRITE, double, 1> acc_x1( regions[0] , FID_VAL );
                const FieldAccessor<READ_ONLY, double, 1>  acc_x3( regions[1] , FID_VAL );

                for ( PointInRectIterator<1> pir(rect) ; pir() ; pir++ )
                    acc_x1[*pir] = op.m_alpha1 * acc_x1[*pir]
                                 + op.m_alpha2 * acc_x3[*pir];
            }
            else
            {
                // Out-of-place mode: x1, x2, x3 are distinct regions
                assert(regions.size() == 3);
                const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1( regions[0] , FID_VAL );
                const FieldAccessor<READ_ONLY, double, 1>     acc_x2( regions[1] , FID_VAL );
                const FieldAccessor<READ_ONLY, double, 1>     acc_x3( regions[2] , FID_VAL );

                for ( PointInRectIterator<1> pir(rect) ; pir() ; pir++ )
                    acc_x1[*pir] = op.m_alpha1 * acc_x2[*pir]
                                 + op.m_alpha2 * acc_x3[*pir];
            }
        }
    };
};

#endif
