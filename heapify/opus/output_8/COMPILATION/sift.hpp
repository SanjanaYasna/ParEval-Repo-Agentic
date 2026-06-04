// sift.hpp — raw-pointer / iterator helpers for heap sift-down
// source: https://github.com/Syntaf/heapify/tree/master
#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstdint>
#include <cstddef>

// Sift a single element down the heap.
// first  – pointer to element 0 of the array
// last   – pointer past the last element (unused; len is authoritative)
// pred   – strict-weak ordering (std::less<T> → max-heap)
// len    – total number of elements
// start  – pointer to the element to sift down
template<typename RandomIt, typename Pred>
void sift_down(RandomIt first, RandomIt /*last*/, Pred pred,
               int64_t len, RandomIt start_it)
{
    int64_t start_idx = start_it - first;
    int64_t child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && pred(*(first + child), *(first + child + 1)))
        ++child;

    if (pred(*(first + child), *(first + start_idx)))
        return;

    auto top = *(first + start_idx);
    int64_t pos = start_idx;
    do {
        *(first + pos) = *(first + child);
        pos = child;

        if ((len - 2) / 2 < child)
            break;

        child = 2 * child + 1;

        if ((child + 1) < len && pred(*(first + child), *(first + child + 1)))
            ++child;
    } while (!pred(*(first + child), top));
    *(first + pos) = top;
}

// Sift down a contiguous range of elements.
// Processes indices corresponding to  start, start-1, …, start-count+1.
template<typename RandomIt, typename Pred>
void sift_down_range(RandomIt first, RandomIt last, Pred pred,
                     int64_t n, RandomIt start, std::size_t count)
{
    for (std::size_t i = 0; i < count; i++) {
        sift_down(first, last, pred, n, start - i);
    }
}

#endif // SIFT_HPP
