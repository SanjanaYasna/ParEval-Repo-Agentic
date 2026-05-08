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
    FID_VAL = 0,
};

// Task IDs for scale_sum2 operations
enum ScaleSum2TaskIDs {
    SCALE_SUM2_TASK_ID = 10,
    SCALE_SUM2_INPLACE_TASK_ID = 11,
};

// Serializable arguments passed to scale_sum2 Legion tasks
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

        // Apply on raw double arrays: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        void operator()(double* x1, const double* x2, const double* x3, size_t N) const
        {
            for (size_t i = 0; i < N; ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }
    };
};

// Legion task: x1 = alpha1 * x2 + alpha2 * x3
// regions[0]: x1 (WRITE_DISCARD), regions[1]: x2 (READ_ONLY), regions[2]: x3 (READ_ONLY)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(regions.size() == 3);
    assert(task->arglen == sizeof(ScaleSum2Args));
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_x3(regions[2], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

// Legion task: x1 = alpha1 * x1 + alpha2 * x2 (in-place variant)
// regions[0]: x1 (READ_WRITE), regions[1]: x2 (READ_ONLY)
inline void scale_sum2_inplace_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    assert(regions.size() == 2);
    assert(task->arglen == sizeof(ScaleSum2Args));
    const ScaleSum2Args &args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_x2(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_x1[*pir] = args.alpha1 * acc_x1[*pir] + args.alpha2 * acc_x2[*pir];
    }
}

#endif
