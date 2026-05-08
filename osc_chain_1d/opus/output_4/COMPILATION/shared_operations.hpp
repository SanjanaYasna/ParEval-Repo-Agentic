// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cassert>
#include "legion.h"

using namespace Legion;

// Field ID for the double-valued state data stored in all state regions.
// Shared across the application (system.hpp, algebra.hpp, odeint.cpp, etc.).
enum FieldIDs {
    FID_VAL = 101,
};

// Serializable POD argument structure for the scale_sum2 Legion task.
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

        // Pack scaling factors into a task argument struct for Legion task launch
        ScaleSum2Args get_args() const
        {
            return ScaleSum2Args{static_cast<double>(m_alpha1),
                                 static_cast<double>(m_alpha2)};
        }
    };
};

// Legion task implementing element-wise: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
//
// Region requirements (set up by the caller / algebra):
//   regions[0] -> x1  (WRITE_DISCARD, field FID_VAL)
//   regions[1] -> x2  (READ_ONLY,     field FID_VAL)
//   regions[2] -> x3  (READ_ONLY,     field FID_VAL)
//
// Task argument: ScaleSum2Args (alpha1, alpha2)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    assert(regions.size() == 3);
    assert(task->arglen == sizeof(ScaleSum2Args));
    const ScaleSum2Args &args =
        *reinterpret_cast<const ScaleSum2Args *>(task->args);

    const FieldAccessor<WRITE_DISCARD, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>     acc_x2(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>     acc_x3(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

#endif
