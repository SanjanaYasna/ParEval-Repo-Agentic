#pragma once

#include <legion.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

namespace heap_legion {

// Field IDs for the heap LogicalRegion
enum HeapFieldID : Legion::FieldID {
  FID_HEAP_VALUE = 0
};

// Task ID for range sift-down work
enum HeapTaskID : Legion::TaskID {
  TID_SIFT_DOWN_RANGE = 1001
};

struct SiftDownRangeArgs {
  std::int64_t len;         // heap length (typically full vector length)
  std::int64_t start_index; // first node index (0-based, heap-relative)
  std::int64_t count;       // number of parent nodes to process backwards
};

// -----------------------------------------------------------------------------
// Iterator-based helpers (kept for compatibility with local/sequential paths)
// -----------------------------------------------------------------------------
template <typename RndIter, typename Pred>
inline void sift_down(RndIter first, RndIter /*last*/, Pred&& pred,
                      typename std::iterator_traits<RndIter>::difference_type len,
                      RndIter start) {
  using difference_type =
      typename std::iterator_traits<RndIter>::difference_type;
  using value_type = typename std::iterator_traits<RndIter>::value_type;

  difference_type child = start - first;
  if (len < 2 || (len - 2) / 2 < child) return;

  child = 2 * child + 1;
  RndIter child_i = first + child;

  if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
    ++child_i;
    ++child;
  }

  if (pred(*child_i, *start)) return;

  value_type top = *start;
  do {
    *start = *child_i;
    start = child_i;

    if ((len - 2) / 2 < child) break;

    child = 2 * child + 1;
    child_i = first + child;

    if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
      ++child_i;
      ++child;
    }
  } while (!pred(*child_i, top));

  *start = top;
}

template <typename RndIter, typename Pred>
inline void sift_down_range(
    RndIter first, RndIter last, Pred&& pred,
    typename std::iterator_traits<RndIter>::difference_type len, RndIter start,
    std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    sift_down<RndIter>(first, last, std::forward<Pred>(pred), len, start - i);
  }
}

// -----------------------------------------------------------------------------
// Legion accessor-based helpers
// -----------------------------------------------------------------------------
template <typename Accessor, typename Pred>
inline void sift_down_accessor(Accessor& acc, Legion::coord_t base,
                               std::int64_t len, std::int64_t start_idx,
                               Pred&& pred) {
  std::int64_t child = start_idx;

  if (len < 2 || (len - 2) / 2 < child) return;

  child = 2 * child + 1;

  auto read_at = [&](std::int64_t idx) -> int {
    return acc[Legion::Point<1>(base + static_cast<Legion::coord_t>(idx))];
  };
  auto write_at = [&](std::int64_t idx, int value) {
    acc[Legion::Point<1>(base + static_cast<Legion::coord_t>(idx))] = value;
  };

  std::int64_t chosen_child = child;
  int child_val = read_at(chosen_child);

  if ((child + 1) < len) {
    const int right_val = read_at(child + 1);
    if (pred(child_val, right_val)) {
      chosen_child = child + 1;
      child_val = right_val;
    }
  }

  if (pred(child_val, read_at(start_idx))) return;

  int top = read_at(start_idx);
  std::int64_t cur = start_idx;

  do {
    write_at(cur, child_val);
    cur = chosen_child;

    if ((len - 2) / 2 < chosen_child) break;

    child = 2 * chosen_child + 1;
    chosen_child = child;
    child_val = read_at(chosen_child);

    if ((child + 1) < len) {
      const int right_val = read_at(child + 1);
      if (pred(child_val, right_val)) {
        chosen_child = child + 1;
        child_val = right_val;
      }
    }
  } while (!pred(child_val, top));

  write_at(cur, top);
}

template <typename Accessor, typename Pred>
inline void sift_down_range_accessor(Accessor& acc, Legion::coord_t base,
                                     std::int64_t len, std::int64_t start_idx,
                                     std::int64_t count, Pred&& pred) {
  for (std::int64_t i = 0; i < count; ++i) {
    sift_down_accessor(acc, base, len, start_idx - i, pred);
  }
}

// -----------------------------------------------------------------------------
// Legion task entry point
// Expects:
//   - task->args is SiftDownRangeArgs
//   - one RW region with field FID_HEAP_VALUE
// -----------------------------------------------------------------------------
inline void sift_down_range_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime) {
  (void)ctx;
  (void)runtime;

  assert(task != nullptr);
  assert(task->arglen == sizeof(SiftDownRangeArgs));
  assert(regions.size() == 1);

  const auto& args = *static_cast<const SiftDownRangeArgs*>(task->args);
  assert(args.count >= 0);
  assert(args.start_index >= 0);
  assert(args.len >= 0);

  Legion::FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_HEAP_VALUE);

  const Legion::coord_t base = 0;
  sift_down_range_accessor(acc, base, args.len, args.start_index, args.count,
                           std::less<int>{});
}

}  // namespace heap_legion
