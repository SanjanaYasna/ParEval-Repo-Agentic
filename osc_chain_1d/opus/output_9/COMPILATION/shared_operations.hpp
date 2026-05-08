// Copyright 2013 Mario Mulansky
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cstddef>
#include <cstring>
#include <cassert>
#include "legion.h"

using namespace Legion;

// Field ID for storing double values in logical regions
enum FieldIDs {
    FID_VAL = 101,
};

// Serializable arguments for the scale_sum2 Legion task
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

        // Apply operation using Legion field accessors over a rect domain
        // (for use inside a Legion task body)
        void operator()(const FieldAccessor<READ_WRITE, double, 1> &x1,
                        const FieldAccessor<READ_ONLY, double, 1>  &x2,
                        const FieldAccessor<READ_ONLY, double, 1>  &x3,
                        const Rect<1> &rect) const
        {
            for (PointInRectIterator<1> pir(rect); pir(); pir++)
            {
                x1[*pir] = m_alpha1 * x2[*pir] + m_alpha2 * x3[*pir];
            }
        }

        // Apply operation on raw double arrays
        // (convenience for when data has already been extracted)
        void operator()(double *x1, const double *x2,
                        const double *x3, size_t n) const
        {
            for (size_t i = 0; i < n; ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }
    };
};

// Standalone Legion task implementation for scale_sum2.
// Region requirements:
//   regions[0] -> x1 (READ_WRITE, FID_VAL)
//   regions[1] -> x2 (READ_ONLY,  FID_VAL)
//   regions[2] -> x3 (READ_ONLY,  FID_VAL)
// Task arguments: ScaleSum2Args {alpha1, alpha2}
// Computes: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(ScaleSum2Args));
    ScaleSum2Args args;
    memcpy(&args, task->args, sizeof(ScaleSum2Args));

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  acc_x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  acc_x3(regions[2], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

#endif
