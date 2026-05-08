// Copyright 2013 Mario Mulansky
// operations functionality for odeint – Legion version
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cstddef>
#include "legion.h"

using namespace Legion;

// Field ID used across all regions storing double values
enum FieldIDs {
    FID_VAL = 0,
};

// Argument structure serialized into Legion task args for scale_sum2
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

        // Inline application on FieldAccessors within an existing task body
        // Computes x1[i] = alpha1 * x2[i] + alpha2 * x3[i]
        void operator()(const FieldAccessor<READ_WRITE, double, 1> &x1,
                        const FieldAccessor<READ_ONLY, double, 1> &x2,
                        const FieldAccessor<READ_ONLY, double, 1> &x3,
                        const Rect<1> &rect) const
        {
            for (PointInRectIterator<1> pir(rect); pir(); pir++)
                x1[*pir] = m_alpha1 * x2[*pir] + m_alpha2 * x3[*pir];
        }

        // Pack coefficients for use as Legion task arguments
        ScaleSum2Args pack_args() const
        {
            return {static_cast<double>(m_alpha1),
                    static_cast<double>(m_alpha2)};
        }

        // Static Legion task body, registered and launched by the algebra.
        //
        // Two modes determined by the number of PhysicalRegions:
        //   2 regions (in-place, x1 aliases x2):
        //     regions[0] x1/x2 (READ_WRITE), regions[1] x3 (READ_ONLY)
        //   3 regions (general case):
        //     regions[0] x1 (READ_WRITE), regions[1] x2 (READ_ONLY),
        //     regions[2] x3 (READ_ONLY)
        static void task_body(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime)
        {
            const ScaleSum2Args *args =
                reinterpret_cast<const ScaleSum2Args *>(task->args);
            const double a1 = args->alpha1;
            const double a2 = args->alpha2;

            Domain dom = runtime->get_index_space_domain(ctx,
                task->regions[0].region.get_index_space());
            Rect<1> rect = dom;

            if (regions.size() == 2)
            {
                // In-place update: x1 == x2
                // x1[i] = a1 * x1[i] + a2 * x3[i]
                const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
                const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[1], FID_VAL);
                for (PointInRectIterator<1> pir(rect); pir(); pir++)
                    acc_x1[*pir] = a1 * acc_x1[*pir] + a2 * acc_x3[*pir];
            }
            else
            {
                // General case: x1[i] = a1 * x2[i] + a2 * x3[i]
                const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
                const FieldAccessor<READ_ONLY,  double, 1> acc_x2(regions[1], FID_VAL);
                const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[2], FID_VAL);
                for (PointInRectIterator<1> pir(rect); pir(); pir++)
                    acc_x1[*pir] = a1 * acc_x2[*pir] + a2 * acc_x3[*pir];
            }
        }
    };
};

#endif
