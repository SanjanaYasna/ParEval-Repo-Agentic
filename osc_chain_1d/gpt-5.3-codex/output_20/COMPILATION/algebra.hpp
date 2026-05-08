// Copyright 2013 Mario Mulansky
// Legion-oriented algebra backend (no HPX dataflow)
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>

struct local_dataflow_algebra
{
    // Applies op element-wise:
    // s1[i] = op( s1[i], s2[i], s3[i] )
    //
    // In the Legion port, this is executed directly in the calling task.
    // Parallelism is expected to come from Legion task decomposition rather
    // than HPX futures/dataflow at this layer.
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
