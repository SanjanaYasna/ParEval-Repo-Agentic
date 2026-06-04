#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

// Runtime-agnostic heap primitives.
// In the Legion version, these are intended to be called from Legion tasks.

template <typename RndIter, typename Pred>
inline void sift_down(
    RndIter first,
    RndIter /*last*/,
    const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start)
{
    using difference_type = typename std::iterator_traits<RndIter>::difference_type;
    using value_type = typename std::iterator_traits<RndIter>::value_type;

    // left-child index of i is 2*i + 1, right-child is 2*i + 2
    difference_type child = start - first;

    if (len < 2 || (len - 2) / 2 < child) {
        return;
    }

    child = 2 * child + 1;
    RndIter child_i = first + child;

    if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
        ++child_i;
        ++child;
    }

    if (pred(*child_i, *start)) {
        return;
    }

    value_type top = *start;
    do {
        *start = *child_i;
        start = child_i;

        if ((len - 2) / 2 < child) {
            break;
        }

        child = 2 * child + 1;
        child_i = first + child;

        if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
            ++child_i;
            ++child;
        }
    } while (!pred(*child_i, top));

    *start = top;
}

template <typename RndIter, typename Pred>
inline void sift_down_range(
    RndIter first,
    RndIter last,
    const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start,
    std::size_t count)
{
    // Apply sift_down to [start, start - count + 1]
    using difference_type = typename std::iterator_traits<RndIter>::difference_type;
    for (std::size_t i = 0; i < count; ++i) {
        sift_down<RndIter>(first, last, pred, len, start - static_cast<difference_type>(i));
    }
}
