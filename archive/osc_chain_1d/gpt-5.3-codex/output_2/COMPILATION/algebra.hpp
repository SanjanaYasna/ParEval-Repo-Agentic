// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <legion.h>

// Legion version of the odeint algebra.
// In this port, element-wise operations are applied directly within the
// currently executing Legion task. Task-level parallelism/dependencies are
// managed outside this algebra helper.
struct local_dataflow_algebra
{
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t N = s1.size();
        assert(s2.size() == N && s3.size() == N);

        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
