#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstddef>
#include <functional>

// ---- Core algorithms (index-based, operate on raw pointer) ----

template<typename T, typename Pred>
void sift_down(T* data, Pred&& pred, long long len, long long start)
{
    long long child = start;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && pred(data[child], data[child + 1])) {
        ++child;
    }

    if (pred(data[child], data[start]))
        return;

    T top = data[start];
    long long pos = start;
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
void sift_down_range(T* data, Pred&& pred, long long len,
                     long long start, std::size_t count)
{
    // Calls sift_down for indices: start, start-1, ..., start-count+1
    for (std::size_t i = 0; i < count; i++) {
        sift_down<T>(data, std::forward<Pred>(pred), len,
                     start - static_cast<long long>(i));
    }
}

#endif // SIFT_HPP
