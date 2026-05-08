// Copyright 2013 Mario Mulansky
// Legion translation of algebra.hpp
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <boost/range/size.hpp>
#include <legion.h>

// In the Legion version, fine-grained HPX dataflow chaining is replaced by
// direct local application inside a Legion task context.
// Parallelism is expected to come from Legion task decomposition at a higher level.
struct local_dataflow_algebra
{
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t N = boost::size(s1);
        assert(boost::size(s2) == N && boost::size(s3) == N);

        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
