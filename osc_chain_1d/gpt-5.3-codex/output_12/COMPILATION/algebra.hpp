// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <legion.h>

// In the Legion version, this algebra is expected to run inside a Legion task.
// Task-level parallelism is handled at the Legion launch level (default mapper),
// while this algebra applies element-wise operations locally.
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
