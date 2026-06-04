#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

// Legion translation note:
// These are local compute kernels (no HPX runtime calls) intended to be invoked
// from Legion tasks operating on contiguous/random-access data ranges.

template <typename RndIter, typename Pred>
inline void sift_down(
    RndIter first,
    [[maybe_unused]] RndIter last,
    const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start)
{
    using difference_type = typename std::iterator_traits<RndIter>::difference_type;
    using value_type = typename std::iterator_traits<RndIter>::value_type;

    // left-child of start is at 2 * start + 1
    // right-child of start is at 2 * start + 2
    difference_type child = start - first;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;
    RndIter child_i = first + child;

    if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
        ++child_i;
        ++child;
    }

    if (pred(*child_i, *start))
        return;

    value_type top = *start;
    do {
        *start = *child_i;
        start = child_i;

        if ((len - 2) / 2 < child)
            break;

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
    // Calls sift_down for a descending range of parent nodes:
    // start, start-1, ..., start-(count-1)
    for (std::size_t i = 0; i < count; ++i) {
        sift_down(first, last, pred, len, start - static_cast<std::ptrdiff_t>(i));
    }
}
