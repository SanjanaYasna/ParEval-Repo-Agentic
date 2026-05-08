// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow algebra to Legion-style eager local algebra.
// Parallelism should be expressed at Legion task granularity (default mapper).
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>

struct local_dataflow_algebra
{
    // Applies op element-wise:
    //   s1[i] = op( s1[i], s2[i], s3[i] )
    //
    // In the Legion translation, this executes eagerly inside the currently
    // running Legion task. Task-level parallelism/data movement is handled
    // by Legion launches elsewhere in the codebase.
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
