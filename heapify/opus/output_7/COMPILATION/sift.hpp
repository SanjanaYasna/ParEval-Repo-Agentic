#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstddef>
#include <utility>

// Sift-down operation for a single node at index start_idx in the array data[0..len-1].
// This is the core heap maintenance operation: if the subtree rooted at start_idx
// violates the heap property (with respect to pred), sift the node down until
// the heap property is restored.
template<typename T, typename Pred>
void sift_down(T* data, Pred&& pred, long long len, long long start_idx)
{
    long long child = start_idx;

    if(len < 2 || (len - 2) / 2 < child)
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
    } while(!pred(data[child], top));
    data[pos] = top;
}

// Calls sift_down for a contiguous range of nodes.
// Starts at start_idx and processes count nodes moving leftward
// (start_idx, start_idx-1, ..., start_idx-count+1).
template<typename T, typename Pred>
void sift_down_range(T* data, Pred&& pred, long long len,
                     long long start_idx, std::size_t count)
{
    for(std::size_t i = 0; i < count; i++) {
        sift_down<T>(data, std::forward<Pred>(pred), len,
                     start_idx - static_cast<long long>(i));
    }
}

#endif // SIFT_HPP
