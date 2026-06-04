#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstdint>

// Core sift-down logic for a single node at position start_idx
// in a max-heap of length len, using less-than comparison (max-heap).
// Operates on a raw int pointer to the heap array.
inline void sift_down(int *first, int64_t len, int64_t start_idx)
{
    int64_t child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && first[child] < first[child + 1])
        ++child;

    if (first[child] < first[start_idx])
        return;

    int top = first[start_idx];
    int64_t pos = start_idx;
    do {
        first[pos] = first[child];
        pos = child;

        if ((len - 2) / 2 < child)
            break;

        child = 2 * child + 1;

        if ((child + 1) < len && first[child] < first[child + 1])
            ++child;

    } while (!(first[child] < top));
    first[pos] = top;
}

// Calls sift_down for `count` consecutive heap node indices:
//   start_idx, start_idx-1, ..., start_idx-count+1
inline void sift_down_range(int *first, int64_t len,
                            int64_t start_idx, int64_t count)
{
    for (int64_t i = 0; i < count; i++) {
        sift_down(first, len, start_idx - i);
    }
}

#endif // SIFT_HPP
