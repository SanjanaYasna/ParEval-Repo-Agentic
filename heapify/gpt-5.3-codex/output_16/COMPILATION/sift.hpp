#pragma once

#include <legion.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <utility>

// ------------------------------------------------------------
// Generic (sequential) sift helpers (kept close to original)
// ------------------------------------------------------------
template <typename RndIter, typename Pred>
inline void sift_down(
    RndIter first, RndIter /*last*/, const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start)
{
  using difference_type = typename std::iterator_traits<RndIter>::difference_type;
  using value_type = typename std::iterator_traits<RndIter>::value_type;

  difference_type child = start - first;

  if (len < 2 || (len - 2) / 2 < child)
    return;

  child = 2 * child + 1;
  RndIter child_i = first + child;

  if ((child + 1) < len && pred(*child_i, *(child_i + 1))) {
    ++child_i;
    ++child;
  }

  if (pred(*child_i, *start))
    return;

  value_type top = *start;
  do {
    *start = *child_i;
    start = child_i;

    if ((len - 2) / 2 < child)
      break;

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
    RndIter first, RndIter last, const Pred& pred,
    typename std::iterator_traits<RndIter>::difference_type len,
    RndIter start, std::size_t count)
{
  using difference_type = typename std::iterator_traits<RndIter>::difference_type;
  for (std::size_t i = 0; i < count; ++i) {
    sift_down(first, last, pred, len, start - static_cast<difference_type>(i));
  }
}

// ------------------------------------------------------------
// Legion task kernel for range-based sift-down over int field
// ------------------------------------------------------------
//
// Expected launch shape:
// - 1 region requirement (READ_WRITE) for heap data field
// - task args = SiftDownRangeTaskArgs
//
struct SiftDownRangeTaskArgs {
  std::int64_t len;       // total heap length
  std::int64_t start;     // starting node index (0-based)
  std::int64_t count;     // number of nodes to process backward from start
  Legion::FieldID fid;    // field containing int heap values
};

inline int heap_get(
    const Legion::FieldAccessor<Legion::READ_WRITE, int, 1>& acc,
    Legion::coord_t base, std::int64_t idx)
{
  return acc[Legion::Point<1>(base + idx)];
}

inline void heap_set(
    const Legion::FieldAccessor<Legion::READ_WRITE, int, 1>& acc,
    Legion::coord_t base, std::int64_t idx, int value)
{
  acc[Legion::Point<1>(base + idx)] = value;
}

inline void sift_down_accessor(
    const Legion::FieldAccessor<Legion::READ_WRITE, int, 1>& acc,
    Legion::coord_t base, std::int64_t len, std::int64_t start)
{
  if (start < 0)
    return;

  std::int64_t child = start;
  if (len < 2 || (len - 2) / 2 < child)
    return;

  child = 2 * child + 1;

  if ((child + 1) < len && heap_get(acc, base, child) < heap_get(acc, base, child + 1)) {
    ++child;
  }

  if (heap_get(acc, base, child) < heap_get(acc, base, start))
    return;

  int top = heap_get(acc, base, start);
  std::int64_t cur = start;

  do {
    heap_set(acc, base, cur, heap_get(acc, base, child));
    cur = child;

    if ((len - 2) / 2 < child)
      break;

    child = 2 * child + 1;
    if ((child + 1) < len && heap_get(acc, base, child) < heap_get(acc, base, child + 1)) {
      ++child;
    }
  } while (!(heap_get(acc, base, child) < top));

  heap_set(acc, base, cur, top);
}

inline void sift_down_range_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
  assert(task != nullptr);
  assert(task->arglen == sizeof(SiftDownRangeTaskArgs));
  assert(regions.size() == 1);

  const auto* args = static_cast<const SiftDownRangeTaskArgs*>(task->args);

  const Legion::FieldAccessor<Legion::READ_WRITE, int, 1> acc(regions[0], args->fid);

  const Legion::Domain dom =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  const Legion::Rect<1> rect = dom.get_rect<1>();
  const Legion::coord_t base = rect.lo[0];

  for (std::int64_t i = 0; i < args->count; ++i) {
    const std::int64_t node = args->start - i;
    if (node < 0)
      break;
    sift_down_accessor(acc, base, args->len, node);
  }
}
