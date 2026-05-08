// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow-style algebra to Legion-style local algebra
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <boost/range/size.hpp>

// In the Legion version, odeint algebra operations are expected to run
// inside Legion tasks. Parallelism is expressed by task decomposition at a
// higher level, so this algebra applies the operation directly per block.
struct local_dataflow_algebra
{
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
