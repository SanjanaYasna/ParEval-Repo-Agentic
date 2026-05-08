// Copyright 2013 Mario Mulansky
// Translated for Legion backend: this file defines local element-wise kernels
// and is runtime-agnostic (task launch is handled elsewhere).
#ifndef SHARED_OPERATIONS_HPP
#define SHARED_OPERATIONS_HPP

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

struct local_dataflow_shared_operations
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

        template <typename S1, typename S2, typename S3>
        S1 operator()(S1 x1, const S2 x2, const S3 x3) const
        {
            assert(x1 && x2 && x3);
            const std::size_t n = x1->size();
            assert(x2->size() == n && x3->size() == n);

            double *out = x1->data();
            const double *in2 = x2->data();
            const double *in3 = x3->data();

            for (std::size_t i = 0; i < n; ++i)
                out[i] = m_alpha1 * in2[i] + m_alpha2 * in3[i];

            return x1;
        }
    };
};

#endif
