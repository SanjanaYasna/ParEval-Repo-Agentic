// Copyright 2013 Mario Mulansky
// Legion-oriented algebra implementation (eager per-block application)
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <boost/range/size.hpp>
#include <legion.h>

// Kept name for compatibility with existing stepper typedefs.
struct local_dataflow_algebra
{
    // In the HPX version this used dataflow futures.
    // In the Legion port, task parallelism/scheduling is handled explicitly
    // in the call sites, so this algebra applies the operation eagerly.
    template <typename S, typename Op>
    void for_each3(S& s1, const S& s2, const S& s3, Op op) const
    {
        const std::size_t N = boost::size(s1);
        assert(boost::size(s2) == N);
        assert(boost::size(s3) == N);

        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
