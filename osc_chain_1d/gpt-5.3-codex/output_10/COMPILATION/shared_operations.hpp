// Copyright 2013 Mario Mulansky
// Translated to a Legion-friendly backend (no HPX dependencies).
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <memory>
#include <cstddef>
#include <cassert>

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// This operations policy is used by boost::numeric::odeint stepper internals.
// In the Legion version, these functors are intended to run inside Legion tasks
// on task-local block data.
struct local_legion_shared_operations
{
    template <typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2() : m_alpha1(0), m_alpha2(0) {}

        scale_sum2(Fac1 alpha1, Fac2 alpha2)
            : m_alpha1(alpha1), m_alpha2(alpha2)
        { }

        template <typename S1, typename S2, typename S3>
        S1 operator()(S1 x1, const S2& x2, const S3& x3) const
        {
            assert(x1 && x2 && x3);
            assert(x1->size() == x2->size());
            assert(x1->size() == x3->size());

            const std::size_t n = x1->size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*x1)[i] = m_alpha1 * (*x2)[i] + m_alpha2 * (*x3)[i];
            }
            return x1;
        }
    };
};

// Backward-compatible alias so existing code using the old HPX-oriented name
// does not need to change immediately.
using local_dataflow_shared_operations = local_legion_shared_operations;

#endif
