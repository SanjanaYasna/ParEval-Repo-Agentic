// Copyright 2013 Mario Mulansky
// operations functionality for odeint – Legion version
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include <cstring>
#include <cassert>
#include "legion.h"

using namespace Legion;

// Common field ID for double-valued data stored in logical regions
enum FieldIDs {
    FID_VAL = 0,
};

// Serializable argument structure for scale_sum2 Legion tasks
struct scale_sum2_args_t {
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

        // Apply on raw double arrays (for use within a Legion task body)
        void operator()(double *x1, const double *x2, const double *x3, std::size_t N) const
        {
            for (std::size_t i = 0; i < N; ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }

        // Pack coefficients into a task-argument struct for Legion task launch
        scale_sum2_args_t pack_args() const
        {
            return { static_cast<double>(m_alpha1),
                     static_cast<double>(m_alpha2) };
        }
    };
};

// Legion task: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]  (3 distinct regions)
// regions[0]: x1 (WRITE_DISCARD, FID_VAL)
// regions[1]: x2 (READ_ONLY,     FID_VAL)
// regions[2]: x3 (READ_ONLY,     FID_VAL)
// task->args : scale_sum2_args_t
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(regions.size() == 3);
    assert(task->arglen == sizeof(scale_sum2_args_t));

    scale_sum2_args_t args;
    std::memcpy(&args, task->args, sizeof(scale_sum2_args_t));

    const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>     acc_x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>     acc_x3(regions[2], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
}

// In-place Legion task: x1[i] = alpha1 * x1[i] + alpha2 * x2[i]  (2 regions)
// regions[0]: x1 (READ_WRITE, FID_VAL)
// regions[1]: x2 (READ_ONLY,  FID_VAL)
// task->args : scale_sum2_args_t
inline void scale_sum2_inplace_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
    assert(regions.size() == 2);
    assert(task->arglen == sizeof(scale_sum2_args_t));

    scale_sum2_args_t args;
    std::memcpy(&args, task->args, sizeof(scale_sum2_args_t));

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  acc_x2(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc_x1[*pir] = args.alpha1 * acc_x1[*pir] + args.alpha2 * acc_x2[*pir];
}

#endif
