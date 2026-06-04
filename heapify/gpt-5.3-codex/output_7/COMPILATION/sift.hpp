#pragma once

#include <cstddef>
#include <cstdint>

namespace heap_legion {

// Max-heap sift-down specialized for int with std::less<int> semantics.
inline void sift_down(int* data, std::ptrdiff_t len, std::ptrdiff_t start_idx) {
  if (data == nullptr || len < 2 || start_idx < 0) return;

  const std::ptrdiff_t last_parent = (len - 2) / 2;
  if (start_idx > last_parent) return;

  std::ptrdiff_t hole = start_idx;
  int top = data[hole];

  while (true) {
    std::ptrdiff_t child = 2 * hole + 1;
    if (child >= len) break;

    if ((child + 1) < len && data[child] < data[child + 1]) {
      ++child;
    }

    if (data[child] < top) break;

    data[hole] = data[child];
    hole = child;

    if (hole > last_parent) break;
  }

  data[hole] = top;
}

inline void sift_down_range(int* data,
                            std::ptrdiff_t len,
                            std::ptrdiff_t start_idx,
                            std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    const std::ptrdiff_t idx = start_idx - static_cast<std::ptrdiff_t>(i);
    if (idx < 0) break;
    sift_down(data, len, idx);
  }
}

} // namespace heap_legion
