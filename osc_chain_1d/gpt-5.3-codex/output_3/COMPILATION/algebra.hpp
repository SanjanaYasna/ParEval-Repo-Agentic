// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow algebra to a Legion-friendly local algebra.
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <legion.h>

// In the Legion version, parallelism is expressed through Legion task launches
// (outside this algebra). The algebra itself applies operations locally.
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
