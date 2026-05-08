// Copyright 2013 Mario Mulansky
// Translated to Legion execution model (no HPX dataflow)
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <boost/range/size.hpp>

struct local_dataflow_algebra
{
    // In the Legion version, dependencies are enforced by task/region ordering.
    // This algebra assumes s1/s2/s3 are already materialized local state blocks.
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t N = boost::size(s1);
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
