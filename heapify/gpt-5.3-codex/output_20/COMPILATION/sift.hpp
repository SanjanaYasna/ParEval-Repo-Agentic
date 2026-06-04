#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>

template <typename RandomIt, typename Pred>
inline void sift_down(RandomIt first, RandomIt last, Pred pred,
                      std::ptrdiff_t len, RandomIt start) {
  std::ptrdiff_t n = std::min<std::ptrdiff_t>(len, last - first);
  if (n <= 1) return;
  if (start < first || start >= first + n) return;

  std::ptrdiff_t hole = start - first;
  if (hole > (n - 2) / 2) return;  // leaf

  using value_type = typename std::iterator_traits<RandomIt>::value_type;
  value_type top = *(first + hole);

  std::ptrdiff_t child = 2 * hole + 1;
  while (child < n) {
    const std::ptrdiff_t right = child + 1;
    if (right < n && pred(*(first + child), *(first + right))) {
      child = right;  // choose larger child for max-heap with std::less
    }

    if (!pred(top, *(first + child))) break;  // top >= child

    *(first + hole) = *(first + child);
    hole = child;
    child = 2 * hole + 1;
  }

  *(first + hole) = top;
}

template <typename RandomIt, typename Pred>
inline void sift_down_range(RandomIt first, RandomIt last, Pred pred,
                            std::ptrdiff_t len, RandomIt start,
                            std::size_t count) {
  std::ptrdiff_t n = std::min<std::ptrdiff_t>(len, last - first);
  if (n <= 1 || count == 0) return;
  if (start < first) return;

  const std::ptrdiff_t start_idx = start - first;
  for (std::size_t i = 0; i < count; ++i) {
    const std::ptrdiff_t idx = start_idx - static_cast<std::ptrdiff_t>(i);
    if (idx < 0) break;
    sift_down(first, last, pred, n, first + idx);
  }
}
