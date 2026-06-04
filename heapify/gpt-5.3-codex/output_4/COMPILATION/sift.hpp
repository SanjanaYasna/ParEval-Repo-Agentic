#pragma once

#include <cstddef>
#include <iterator>

// Compute kernel used by Legion tasks (runtime-agnostic).
template <typename RndIter, typename Pred>
inline void sift_down(
    RndIter first, RndIter last, const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start)
{
    (void)last; // kept for signature compatibility with existing call sites

    using difference_type = typename std::iterator_traits<RndIter>::difference_type;
    using value_type = typename std::iterator_traits<RndIter>::value_type;

    // left child index  = 2 * parent + 1
    // right child index = 2 * parent + 2
    difference_type child = start - first;

    if (len < 2 || (len - 2) / 2 < child) {
        return;
    }

    child = 2 * child + 1;
    RndIter child_i = first + child;

    // Select larger child for max-heap when pred is std::less<T>
    if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
        ++child_i;
        ++child;
    }

    // Heap property already satisfied
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
    RndIter first, RndIter last, const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start, std::size_t count)
{
    using difference_type = typename std::iterator_traits<RndIter>::difference_type;

    // Apply sift_down to a descending contiguous range:
    // [start, start-1, ..., start-(count-1)]
    for (std::size_t i = 0; i < count; ++i) {
        sift_down(first, last, pred, len, start - static_cast<difference_type>(i));
    }
}
