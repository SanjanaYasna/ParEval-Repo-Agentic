// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow algebra to Legion-compatible eager algebra.
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <legion.h>

struct local_dataflow_algebra
{
    // In the Legion port, this algebra runs eagerly inside a Legion task.
    // Parallelism should be expressed by the caller (e.g., launching index
    // tasks over blocks) rather than by HPX futures/dataflow here.
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        assert(s1.size() == s2.size());
        assert(s1.size() == s3.size());

        const std::size_t N = s1.size();
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
