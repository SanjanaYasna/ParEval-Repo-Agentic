// sift.hpp — Legion version
// source: translated from https://github.com/Syntaf/heapify/tree/master
#ifndef SIFT_HPP
#define SIFT_HPP

#include <functional>
#include <cstdint>
#include <vector>
#include <cassert>

// ----------------------------------------------------------------
// Core computational helpers (pointer + index based)
// ----------------------------------------------------------------

// sift_down: restore the heap property for the sub-tree rooted at
// data[start_idx] inside an array of length 'len'.
// Pred should be std::less<T> for a max-heap.
template<typename T, typename Pred>
void sift_down(T* data, Pred&& pred, int64_t len, int64_t start_idx)
{
    int64_t child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    // left child
    child = 2 * child + 1;

    // pick the larger child
    if ((child + 1) < len && pred(data[child], data[child + 1])) {
        ++child;
    }

    // heap property already satisfied
    if (pred(data[child], data[start_idx]))
        return;

    T   top = data[start_idx];
    int64_t pos = start_idx;

    do {
        data[pos] = data[child];
        pos = child;

        if ((len - 2) / 2 < child)
            break;

        child = 2 * child + 1;

        if ((child + 1) < len && pred(data[child], data[child + 1])) {
            ++child;
        }
    } while (!pred(data[child], top));

    data[pos] = top;
}

// sift_down_range: call sift_down for 'count' consecutive nodes
// starting at start_idx and moving towards smaller indices.
template<typename T, typename Pred>
void sift_down_range(T* data, Pred&& pred, int64_t len,
                     int64_t start_idx, int64_t count)
{
    for (int64_t i = 0; i < count; ++i) {
        sift_down<T>(data, std::forward<Pred>(pred), len, start_idx - i);
    }
}

#endif // SIFT_HPP
