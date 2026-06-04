#pragma once

#include "legion.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>

namespace heap_legion {

using namespace Legion;

// Keep task IDs centralized for Legion task launch/registration.
constexpr TaskID SIFT_DOWN_RANGE_TASK_ID = 1002;

// Arguments passed to each sift-down range Legion task.
struct SiftDownRangeArgs {
  std::int64_t len;    // heap length
  std::int64_t start;  // start node index (0-based heap index)
  std::int64_t count;  // number of consecutive nodes to process: start, start-1, ...
};

template <typename Accessor, typename T>
inline T load_elem(const Accessor& acc, coord_t base, std::int64_t idx) {
  return acc[Point<1>(base + static_cast<coord_t>(idx))];
}

template <typename Accessor, typename T>
inline void store_elem(Accessor& acc, coord_t base, std::int64_t idx, const T& value) {
  acc[Point<1>(base + static_cast<coord_t>(idx))] = value;
}

template <typename Accessor, typename T, typename Pred>
inline void sift_down(Accessor& acc, coord_t base, std::int64_t len, std::int64_t start, Pred pred) {
  // No children
  if (len < 2 || ((len - 2) / 2) < start) return;

  std::int64_t child = 2 * start + 1;

  // Pick larger child for max-heap behavior with std::less
  if ((child + 1) < len &&
      pred(load_elem<Accessor, T>(acc, base, child), load_elem<Accessor, T>(acc, base, child + 1))) {
    ++child;
  }

  // Heap property already satisfied
  if (pred(load_elem<Accessor, T>(acc, base, child), load_elem<Accessor, T>(acc, base, start))) return;

  T top = load_elem<Accessor, T>(acc, base, start);

  do {
    store_elem<Accessor, T>(acc, base, start, load_elem<Accessor, T>(acc, base, child));
    start = child;

    if (((len - 2) / 2) < child) break;

    child = 2 * child + 1;

    if ((child + 1) < len &&
        pred(load_elem<Accessor, T>(acc, base, child), load_elem<Accessor, T>(acc, base, child + 1))) {
      ++child;
    }
  } while (!pred(load_elem<Accessor, T>(acc, base, child), top));

  store_elem<Accessor, T>(acc, base, start, top);
}

template <typename Accessor, typename T, typename Pred>
inline void sift_down_range(Accessor& acc, coord_t base, std::int64_t len,
                            std::int64_t start, std::int64_t count, Pred pred) {
  for (std::int64_t i = 0; i < count; ++i) {
    const std::int64_t idx = start - i;
    if (idx < 0) break;
    sift_down<Accessor, T>(acc, base, len, idx, pred);
  }
}

// Legion task variant: runs sift_down_range on one chunk.
inline void sift_down_range_task(const Task* task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx,
                                 Runtime* runtime) {
  assert(task != nullptr);
  assert(runtime != nullptr);
  assert(regions.size() == 1);
  assert(task->arglen == static_cast<int>(sizeof(SiftDownRangeArgs)));
  assert(!task->regions.empty());
  assert(!task->regions[0].instance_fields.empty());

  const auto* args = static_cast<const SiftDownRangeArgs*>(task->args);

  const Domain dom =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  const Rect<1> bounds = dom;
  const coord_t base = bounds.lo[0];

  const FieldID fid = task->regions[0].instance_fields[0];
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], fid);

  sift_down_range<decltype(acc), int>(
      acc, base, args->len, args->start, args->count, std::less<int>{});
}

// Call this once during startup (before Runtime::start).
inline void preregister_sift_tasks() {
  TaskVariantRegistrar registrar(SIFT_DOWN_RANGE_TASK_ID, "sift_down_range_task");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf();
  Runtime::preregister_task_variant<sift_down_range_task>(registrar, "sift_down_range_task");
}

}  // namespace heap_legion
