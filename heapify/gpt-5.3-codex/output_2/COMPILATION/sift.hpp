#pragma once

#include <legion.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

namespace heap_legion {

// Field used for heap values in the LogicalRegion.
enum : Legion::FieldID { FID_HEAP_VALUE = 0 };

// Task ID for range-based sift work (can be used by heaps.cpp task registration/launch).
enum : Legion::TaskID { SIFT_DOWN_RANGE_TASK_ID = 2 };

struct SiftDownRangeTaskArgs {
  std::int64_t len;    // total heap length
  std::int64_t start;  // starting node index (inclusive)
  std::int64_t count;  // number of nodes to process backwards: start, start-1, ...
};

// -----------------------------------------------------------------------------
// Generic iterator-based helpers (execution-model agnostic, callable in a task)
// -----------------------------------------------------------------------------
template <typename RndIter, typename Pred>
inline void sift_down(RndIter first, RndIter last, Pred pred,
                      typename std::iterator_traits<RndIter>::difference_type len,
                      RndIter start) {
  (void)last;
  using difference_type = typename std::iterator_traits<RndIter>::difference_type;
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
inline void sift_down_range(RndIter first, RndIter last, Pred pred,
                            typename std::iterator_traits<RndIter>::difference_type len,
                            RndIter start, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    sift_down(first, last, pred, len, start - static_cast<std::ptrdiff_t>(i));
  }
}

// -----------------------------------------------------------------------------
// Legion accessor-based helper + task entry point
// -----------------------------------------------------------------------------
template <typename Pred>
inline void sift_down_indexed(Legion::FieldAccessor<READ_WRITE, int, 1>& acc,
                              std::int64_t base, std::int64_t len,
                              std::int64_t start_idx, Pred pred) {
  auto p = [base](std::int64_t idx) -> Legion::Point<1> {
    return Legion::Point<1>(base + idx);
  };

  std::int64_t child = start_idx;
  if (len < 2 || (len - 2) / 2 < child) return;

  child = 2 * child + 1;

  if ((child + 1) < len && pred(acc[p(child)], acc[p(child + 1)])) {
    ++child;
  }

  if (pred(acc[p(child)], acc[p(start_idx)])) return;

  int top = acc[p(start_idx)];
  std::int64_t cur = start_idx;

  do {
    acc[p(cur)] = acc[p(child)];
    cur = child;

    if ((len - 2) / 2 < child) break;

    child = 2 * child + 1;
    if ((child + 1) < len && pred(acc[p(child)], acc[p(child + 1)])) {
      ++child;
    }
  } while (!pred(acc[p(child)], top));

  acc[p(cur)] = top;
}

inline void sift_down_range_task(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& regions,
                                 Legion::Context ctx,
                                 Legion::Runtime* runtime) {
  assert(task != nullptr);
  assert(task->arglen == sizeof(SiftDownRangeTaskArgs));
  assert(regions.size() == 1);

  const auto* args = static_cast<const SiftDownRangeTaskArgs*>(task->args);

  Legion::FieldAccessor<READ_WRITE, int, 1> values(regions[0], FID_HEAP_VALUE);

  const Legion::Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  const std::int64_t base = rect.lo[0];
  const std::int64_t rect_len = rect.hi[0] - rect.lo[0] + 1;
  const std::int64_t len = std::min<std::int64_t>(args->len, rect_len);

  if (len <= 1 || args->count <= 0) return;

  const std::int64_t start = std::min<std::int64_t>(args->start, len - 1);
  const std::int64_t actual_count = std::min<std::int64_t>(args->count, start + 1);

  for (std::int64_t i = 0; i < actual_count; ++i) {
    sift_down_indexed(values, base, len, start - i, std::less<int>{});
  }
}

}  // namespace heap_legion
