#ifndef SIFT_HPP
#define SIFT_HPP

#include <legion.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>

namespace heap_legion {

// Field ID for heap values in the logical region.
enum HeapFieldID : Legion::FieldID {
  FID_HEAP_VALUE = 0
};

// Task ID for per-chunk sift-down work (register in heaps.cpp).
static constexpr Legion::TaskID SIFT_DOWN_RANGE_TASK_ID = 1001;

// Arguments passed to each sift-down range task.
struct SiftDownRangeArgs {
  Legion::coord_t start_index;   // absolute index in the region domain
  std::size_t count;             // number of parent nodes to process (descending)
  Legion::coord_t heap_length;   // logical heap length (usually full domain length)
};

template <typename Pred>
inline void sift_down(
    const Legion::FieldAccessor<Legion::READ_WRITE, int, 1>& acc,
    Legion::coord_t lo,
    Legion::coord_t len,
    Legion::coord_t start,
    Pred pred)
{
  // Convert absolute index to heap-relative index.
  Legion::coord_t child = start - lo;

  if (len < 2 || ((len - 2) / 2) < child)
    return;

  auto read_at = [&](Legion::coord_t idx) -> int {
    return acc[Legion::Point<1>(idx)];
  };
  auto write_at = [&](Legion::coord_t idx, int v) {
    acc[Legion::Point<1>(idx)] = v;
  };

  child = 2 * child + 1;
  Legion::coord_t child_idx = lo + child;

  if ((child + 1) < len && pred(read_at(child_idx), read_at(child_idx + 1))) {
    ++child_idx;
    ++child;
  }

  if (pred(read_at(child_idx), read_at(start)))
    return;

  int top = read_at(start);
  do {
    write_at(start, read_at(child_idx));
    start = child_idx;

    if (((len - 2) / 2) < child)
      break;

    child = 2 * child + 1;
    child_idx = lo + child;

    if ((child + 1) < len && pred(read_at(child_idx), read_at(child_idx + 1))) {
      ++child_idx;
      ++child;
    }
  } while (!pred(read_at(child_idx), top));

  write_at(start, top);
}

template <typename Pred>
inline void sift_down_range(
    const Legion::FieldAccessor<Legion::READ_WRITE, int, 1>& acc,
    Legion::coord_t lo,
    Legion::coord_t len,
    Legion::coord_t start,
    std::size_t count,
    Pred pred)
{
  // Calls sift_down for a descending range: start, start-1, ...
  for (std::size_t i = 0; i < count; ++i) {
    Legion::coord_t idx = start - static_cast<Legion::coord_t>(i);
    if (idx < lo) break;
    sift_down(acc, lo, len, idx, pred);
  }
}

// Legion task variant to process one chunk/range of sift-down work.
inline void sift_down_range_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
  assert(task != nullptr);
  assert(regions.size() == 1);
  assert(task->arglen == sizeof(SiftDownRangeArgs));

  const auto* args = static_cast<const SiftDownRangeArgs*>(task->args);

  const Legion::FieldAccessor<Legion::READ_WRITE, int, 1> acc(
      regions[0], FID_HEAP_VALUE);

  Legion::Domain dom = runtime->get_index_space_domain(
      ctx, regions[0].get_logical_region().get_index_space());
  Legion::Rect<1> rect = dom.get_rect<1>();

  const Legion::coord_t lo = rect.lo[0];
  const Legion::coord_t dom_len = rect.hi[0] - rect.lo[0] + 1;
  const Legion::coord_t len =
      (args->heap_length > 0) ? std::min(args->heap_length, dom_len) : dom_len;

  sift_down_range(acc, lo, len, args->start_index, args->count, std::less<int>{});
}

} // namespace heap_legion

#endif // SIFT_HPP
