// Copyright 2013 Mario Mulansky
// Translated to Legion execution model
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include "legion.h"

using namespace Legion;

// Field ID used across the codebase for storing double data in logical regions
enum FieldIDs {
    FID_X = 101,
};

// Task IDs for stepper operations
enum OperationTaskIDs {
    SCALE_SUM2_TASK_ID = 10,
};

// Serializable arguments for the scale_sum2 Legion task
struct scale_sum2_args_t {
    double alpha1;
    double alpha2;
};

// Legion task: x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
// regions[0]: x1 (READ_WRITE, FID_X)
// regions[1]: x2 (READ_ONLY,  FID_X)
// regions[2]: x3 (READ_ONLY,  FID_X)
inline void scale_sum2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime)
{
    const scale_sum2_args_t &args =
        *reinterpret_cast<const scale_sum2_args_t *>(task->args);

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,  double, 1> acc_x2(regions[1], FID_X);
    const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[2], FID_X);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_x1[*pir] = args.alpha1 * acc_x2[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

struct local_dataflow_shared_operations
{
    template<typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        Fac1 m_alpha1;
        Fac2 m_alpha2;

        scale_sum2()
            : m_alpha1(0), m_alpha2(0)
        {}

        scale_sum2(Fac1 alpha1, Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2)
        {}

        // Return the Legion task ID for this operation
        static constexpr TaskID task_id() { return SCALE_SUM2_TASK_ID; }

        // Pack factors into a serializable struct for TaskArgument
        scale_sum2_args_t get_args() const
        {
            return { static_cast<double>(m_alpha1),
                     static_cast<double>(m_alpha2) };
        }

        // Direct application on raw double arrays (for inline use within a task)
        void operator()(double *x1, const double *x2, const double *x3, size_t N) const
        {
            for (size_t i = 0; i < N; ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }
    };
};

#endif
