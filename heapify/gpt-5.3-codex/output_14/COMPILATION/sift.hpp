#ifndef SIFT_HPP
#define SIFT_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>

// -----------------------------------------------------------------------------
// Generic local sift-down helpers (used inside Legion CPU tasks)
// -----------------------------------------------------------------------------
template <typename RndIter, typename Pred>
inline void sift_down(
    RndIter first, RndIter last, const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start)
{
    using difference_type =
        typename std::iterator_traits<RndIter>::difference_type;
    using value_type = typename std::iterator_traits<RndIter>::value_type;

    (void)last; // Kept for API compatibility with original code.

    difference_type child = start - first;
    if (len < 2 || child < 0 || (len - 2) / 2 < child) return;

    child = 2 * child + 1;
    RndIter child_i = first + child;

    if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
        ++child_i;
        ++child;
    }

    if (pred(*child_i, *start)) return;

    value_type top = *start;
    do {
        *start = *child_i;
        start = child_i;

        if ((len - 2) / 2 < child) break;

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
    using difference_type =
        typename std::iterator_traits<RndIter>::difference_type;

    const difference_type start_idx = start - first;
    if (start_idx < 0) return;

    for (std::size_t i = 0; i < count; ++i) {
        const difference_type offset = static_cast<difference_type>(i);
        if (offset > start_idx) break;
        sift_down(first, last, pred, len, start - offset);
    }
}

#endif // SIFT_HPP
