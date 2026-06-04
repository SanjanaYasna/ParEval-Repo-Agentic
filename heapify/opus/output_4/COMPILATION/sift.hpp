// sift.hpp
// Translated from HPX iterator-based interface to index-based interface
// suitable for use with Legion FieldAccessor-derived pointers.
#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstddef>
#include <iterator>

template<typename T, typename Pred>
void sift_down(T* data, long long len, Pred&& pred, long long start_idx)
{
    // left-child of start_idx is at 2 * start_idx + 1
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
    } while (!pred(data[child], top));
    data[pos] = top;
}

template<typename T, typename Pred>
void sift_down_range(T* data, long long len, Pred&& pred,
                     long long start_idx, std::size_t count)
{
    // Simply calls sift_down for a range of elements,
    // processing indices start_idx, start_idx-1, ..., start_idx-(count-1)
    for (std::size_t i = 0; i < count; i++) {
        sift_down<T>(data, len, std::forward<Pred>(pred),
                     start_idx - static_cast<long long>(i));
    }
}

#endif // SIFT_HPP
