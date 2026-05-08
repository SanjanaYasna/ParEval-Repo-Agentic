// Copyright 2013 Mario Mulansky
// Translated from HPX-style shared operations to Legion-friendly local operations.
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cstddef>
#include <cassert>

using dvec = std::vector<double>;

struct local_legion_shared_operations
{
    template <typename Fac1, typename Fac2 = Fac1>
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

        // Legion version: operate directly on local block data (no futures/dataflow).
        template <typename S1, typename S2, typename S3>
        void operator()(S1 &x1, const S2 &x2, const S3 &x3) const
        {
            const std::size_t n = x1.size();
            assert(x2.size() == n && x3.size() == n);

            for (std::size_t i = 0; i < n; ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];
        }
    };
};

// Backward-compatible alias to minimize changes in other translated files.
using local_dataflow_shared_operations = local_legion_shared_operations;

#endif
