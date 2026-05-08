// Translated from HPX to Legion-oriented execution style.
// The algebra now applies operations directly inside the currently running Legion task.

#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include <cstddef>
#include <cassert>
#include <legion.h>

struct local_dataflow_algebra
{
    // Applies op element-wise:
    //   s1[i] = op(s1[i], s2[i], s3[i])
    //
    // In the Legion version, this runs synchronously within the caller task.
    // Parallelism is expected to come from Legion task decomposition elsewhere.
    template <typename S, typename Op>
    void for_each3(S& s1, const S& s2, const S& s3, Op op) const
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
