// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <legion.h>

// Legion version of the algebra backend.
//
// In the HPX version, each element update was launched via dataflow futures.
// In the Legion port, element-wise updates are applied eagerly within the
// currently running Legion task. Parallelism should be expressed at the task
// launch level (e.g., index launches over blocks), using Legion's default mapper.
struct local_dataflow_algebra
{
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t N = s1.size();
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
