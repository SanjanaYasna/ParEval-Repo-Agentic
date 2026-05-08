// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include <cassert>
#include <vector>
#include "legion.h"

using namespace Legion;

// Common field ID used across the codebase for double-valued fields
enum FieldIDs {
    FID_VAL = 0,
};

// Task IDs used across the codebase
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_BLOCK_TASK_ID,
    SYSTEM_CENTER_BLOCK_TASK_ID,
    SYSTEM_LAST_BLOCK_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    ENERGY_TASK_ID,
};

// Plain-old-data struct for passing scale_sum2 coefficients via TaskArgument
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

        // Apply element-wise on raw double arrays (usable inside any task body)
        void apply( double *x1 , const double *x2 , const double *x3 , std::size_t n ) const
        {
            for( std::size_t i = 0 ; i < n ; ++i )
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }

        // Build a POD args struct suitable for use as a TaskArgument
        ScaleSum2Args get_args() const
        {
            return ScaleSum2Args{ static_cast<double>(m_alpha1),
                                  static_cast<double>(m_alpha2) };
        }
    };
};

// Legion task implementation for scale_sum2
// regions[0]: x1 (READ_WRITE) — destination, also serves as first source
// regions[1]: x2 (READ_ONLY)  — first source weighted by alpha1
// regions[2]: x3 (READ_ONLY)  — second source weighted by alpha2
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(regions.size() == 3);
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> x3(regions[2], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
        x1[*pir] = args.alpha1 * x2[*pir] + args.alpha2 * x3[*pir];
    }
}

#endif
