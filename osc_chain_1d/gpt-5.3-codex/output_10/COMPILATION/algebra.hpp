// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>

// Legion translation note:
// In HPX this algebra used dataflow futures for dependency tracking.
// In Legion, dependencies are typically expressed through task launches
// and region requirements, so this algebra performs the local element-wise
// operation eagerly inside the executing task.
struct local_dataflow_algebra
{
    // for now just a single state type
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        assert(s1.size() == s2.size());
        assert(s1.size() == s3.size());

        const std::size_t N = s1.size();
        for (std::size_t i = 0; i < N; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
