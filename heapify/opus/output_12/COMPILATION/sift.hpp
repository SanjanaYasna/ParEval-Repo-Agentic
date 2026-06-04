// source: translated from https://github.com/Syntaf/heapify/tree/master
#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstddef>
#include <functional>

// Core sift_down: operates on a raw pointer with index-based access.
// Equivalent to the original iterator-based version where
//   data   <-> first (base iterator)
//   len    <-> len   (distance from first to last)
//   start_idx <-> (start - first)
template<typename T, typename Pred>
void sift_down(T* data, long long len, long long start_idx, Pred&& pred)
{
    // left-child  of start_idx is at 2 * start_idx + 1
    // right-child of start_idx is at 2 * start_idx + 2

    long long child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && pred(data[child], data[child + 1])) {
        ++child;
    }

    if (pred(data[child], data[start_idx]))
        return;

    T top = data[start_idx];
    long long pos = start_idx;
    do {
        data[pos] = data[child];
        pos = child;

        if ((len - 2) / 2 < child)
            break;

        child = 2 * child + 1;

        if ((child + 1) < len && pred(data[child], data[child + 1])) {
            ++child;
        }
    } while (!(pred(data[child], top)));
    data[pos] = top;
}

// Calls sift_down for a range of consecutive elements, starting at
// start_idx and moving leftward for count nodes.
template<typename T, typename Pred>
void sift_down_range(T* data, long long len, long long start_idx,
                     std::size_t count, Pred&& pred)
{
    for (std::size_t i = 0; i < count; i++) {
        sift_down<T>(data, len, start_idx - static_cast<long long>(i),
                     std::forward<Pred>(pred));
    }
}

#endif // SIFT_HPP
