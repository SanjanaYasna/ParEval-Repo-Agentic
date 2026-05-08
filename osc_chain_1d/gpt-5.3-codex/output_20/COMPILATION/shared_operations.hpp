// Copyright 2013 Mario Mulansky
// Translated from HPX-oriented shared state ops to Legion-friendly local ops.
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <vector>
#include <cstddef>
#include <cassert>

typedef std::vector<double> dvec;

struct local_legion_shared_operations
{
    template <typename Fac1, typename Fac2 = Fac1>
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2() : m_alpha1(0), m_alpha2(0) {}
        scale_sum2(Fac1 alpha1, Fac2 alpha2) : m_alpha1(alpha1), m_alpha2(alpha2) {}

        // Works on local block containers (e.g., std::vector<double>) inside Legion tasks.
        // Returns x1 by reference so it can be used in either expression-style or statement-style code.
        template <typename S1, typename S2, typename S3>
        S1& operator()(S1& x1, const S2& x2, const S3& x3) const
        {
            assert(x1.size() == x2.size());
            assert(x1.size() == x3.size());

            for (std::size_t i = 0; i < x1.size(); ++i)
                x1[i] = m_alpha1 * x2[i] + m_alpha2 * x3[i];

            return x1;
        }
    };
};

// Backward-compatible alias to minimize required changes in other translated files.
using local_dataflow_shared_operations = local_legion_shared_operations;

#endif
