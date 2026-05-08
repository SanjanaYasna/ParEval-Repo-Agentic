// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow algebra to Legion-compatible algebra
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <iterator>

struct local_dataflow_algebra
{
    // In the Legion version, this algebra is executed inside Legion tasks.
    // We apply the operation directly for each block entry.
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        using std::size;
        const std::size_t N = size(s1);
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
