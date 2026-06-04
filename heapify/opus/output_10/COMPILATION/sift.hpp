// sift.hpp - Legion execution model version
#ifndef SIFT_HPP
#define SIFT_HPP

#include <cstddef>
#include <cstdint>
#include <functional>

// Sifts down a single element at index `start_idx` within an array of length `len`.
// `data` points to the beginning of the array (replaces the `first` iterator).
// `pred` is the comparison predicate (e.g. std::less<int>() for a max-heap).
template<typename T, typename Pred>
void sift_down(T* data, Pred&& pred, std::ptrdiff_t len, std::ptrdiff_t start_idx)
{
    // left-child of start_idx is at 2 * start_idx + 1
    // right-child of start_idx is at 2 * start_idx + 2

    std::ptrdiff_t child = start_idx;

    if (len < 2 || (len - 2) / 2 < child)
        return;

    child = 2 * child + 1;

    if ((child + 1) < len && pred(data[child], data[child + 1])) {
        ++child;
    }

    if (pred(data[child], data[start_idx]))
        return;

    T top = data[start_idx];
    std::ptrdiff_t pos = start_idx;
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

// Calls sift_down for `count` consecutive elements starting at index `start_idx`
// and working downward (start_idx, start_idx-1, ..., start_idx-count+1).
// `data` points to the beginning of the array.
// `len` is the total number of elements in the array.
template<typename T, typename Pred>
void sift_down_range(T* data, Pred&& pred, std::ptrdiff_t len,
                     std::ptrdiff_t start_idx, std::size_t count)
{
    for (std::size_t i = 0; i < count; i++) {
        sift_down<T>(data, std::forward<Pred>(pred), len,
                     start_idx - static_cast<std::ptrdiff_t>(i));
    }
}

#endif // SIFT_HPP
