// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <boost/range/size.hpp>

// Legion port note:
// Block-level parallelism should be expressed by Legion task/index-task launches
// in the caller. This algebra remains a local element-wise transform usable inside
// a Legion task.
struct local_dataflow_algebra
{
    template <typename S, typename Op>
    inline void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t N = boost::size(s1);
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
