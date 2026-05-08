// Copyright 2013 Mario Mulansky
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>

// Legion port:
// In the HPX version, each element update was wrapped in dataflow/futures.
// In the Legion version, this algebra is executed eagerly inside the currently
// running Legion task. Coarse-grained parallelism is handled by Legion tasking
// in the surrounding code.
struct local_dataflow_algebra
{
    // For now just a single state type, matching original behavior.
    template <typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op) const
    {
        const std::size_t n = s1.size();
        assert(s2.size() == n && s3.size() == n);

        for (std::size_t i = 0; i < n; ++i)
        {
            s1[i] = op(s1[i], s2[i], s3[i]);
        }
    }
};

#endif
