// sift.hpp — pure algorithmic helpers (no Legion dependency)
// source: https://github.com/Syntaf/heapify/tree/master
#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstdint>
#include <functional>

// Core sift-down operating on a raw pointer and an index.
template<typename T, typename Pred>
void sift_down(T* data, int64_t len, int64_t start_idx, Pred&& pred)
{
    int64_t child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && pred(data[child], data[child + 1])) {
        ++child;
    }

    if (pred(data[child], data[start_idx]))
        return;

    T top = data[start_idx];
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

// Calls sift_down for a contiguous range of nodes:
//   start_idx, start_idx-1, ..., start_idx-count+1
template<typename T, typename Pred>
void sift_down_range(T* data, int64_t len, int64_t start_idx,
                     int64_t count, Pred&& pred)
{
    for (int64_t i = 0; i < count; i++) {
        sift_down<T>(data, len, start_idx - i, std::forward<Pred>(pred));
    }
}

#endif // SIFT_HPP
