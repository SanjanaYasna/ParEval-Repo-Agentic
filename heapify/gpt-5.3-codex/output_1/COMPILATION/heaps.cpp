// Translated from HPX to Legion execution model (default mapper)
#include "legion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_LEVEL_TASK_ID
};

enum FieldIDs {
  FID_VALUE = 1
};

struct Options {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
};

struct LevelTaskArgs {
  std::size_t n;           // total vector size
  std::size_t level_start; // right-most parent index for this level
  std::size_t chunk_size;  // nodes per task
  std::size_t items;       // number of parent nodes in this level
};

static std::size_t parse_size_or_default(const char* s, std::size_t def) {
  if (s == nullptr) return def;
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) return def;
  return static_cast<std::size_t>(v);
}

static Options parse_options(const InputArgs& input_args) {
  Options opts;
  for (int i = 1; i < input_args.argc; ++i) {
    std::string arg(input_args.argv[i]);

    if (arg == "--vector_size" && (i + 1) < input_args.argc) {
      opts.vector_size = parse_size_or_default(input_args.argv[++i], opts.vector_size);
    } else if (arg.rfind("--vector_size=", 0) == 0) {
      opts.vector_size = parse_size_or_default(arg.c_str() + std::string("--vector_size=").size(), opts.vector_size);
    } else if (arg == "--chunk_size" && (i + 1) < input_args.argc) {
      opts.chunk_size = parse_size_or_default(input_args.argv[++i], opts.chunk_size);
    } else if (arg.rfind("--chunk_size=", 0) == 0) {
      opts.chunk_size = parse_size_or_default(arg.c_str() + std::string("--chunk_size=").size(), opts.chunk_size);
    }
  }
  return opts;
}

static inline std::size_t rightmost_parent_of_prev_level(std::size_t start) {
  // Given the right-most node index of the current level, return the
  // right-most node index of the previous level.
  // Node level boundaries are:
  //   level l: [2^l - 1, 2^(l+1) - 2]
  if (start == 0) return 0;

  const std::size_t x = start + 1; // shift to [2^l, 2^(l+1)-1]
  std::size_t p = 1;
  while (p <= (x >> 1)) p <<= 1; // largest power-of-two <= x
  const std::size_t level_left = p - 1;
  return (level_left == 0) ? 0 : (level_left - 1);
}

template <typename Accessor>
static inline void sift_down_raw(Accessor& acc, std::size_t len, std::size_t start_idx) {
  // Max-heap sift-down with pred=std::less<int>
  if (len < 2) return;
  if (((len - 2) / 2) < start_idx) return; // no children

  std::size_t child = 2 * start_idx + 1;
  std::size_t child_i = child;

  if ((child + 1) < len) {
    int left_v = acc[Point<1>(static_cast<coord_t>(child))];
    int right_v = acc[Point<1>(static_cast<coord_t>(child + 1))];
    if (left_v < right_v) {
      child_i = child + 1;
      child = child + 1;
    }
  }

  int child_val = acc[Point<1>(static_cast<coord_t>(child_i))];
  int start_val = acc[Point<1>(static_cast<coord_t>(start_idx))];
  if (child_val < start_val) return;

  int top = start_val;
  std::size_t cur = start_idx;

  do {
    acc[Point<1>(static_cast<coord_t>(cur))] =
        acc[Point<1>(static_cast<coord_t>(child_i))];
    cur = child_i;

    if (((len - 2) / 2) < child) break; // no more children

    child = 2 * child + 1;
    child_i = child;

    if ((child + 1) < len) {
      int l = acc[Point<1>(static_cast<coord_t>(child))];
      int r = acc[Point<1>(static_cast<coord_t>(child + 1))];
      if (l < r) {
        child_i = child + 1;
        child = child + 1;
      }
    }
  } while (!(acc[Point<1>(static_cast<coord_t>(child_i))] < top));

  acc[Point<1>(static_cast<coord_t>(cur))] = top;
}

template <typename Accessor>
static inline void sift_down_range_raw(Accessor& acc, std::size_t len, std::size_t start_idx, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (start_idx < i) break;
    sift_down_raw(acc, len, start_idx - i);
  }
}

void sift_level_task(const Task* task,
                     const std::vector<PhysicalRegion>& regions,
                     Context /*ctx*/,
                     Runtime* /*runtime*/) {
  const LevelTaskArgs* args = reinterpret_cast<const LevelTaskArgs*>(task->args);
  if (args == nullptr || regions.empty()) return;

  const std::size_t chunk_id = static_cast<std::size_t>(task->index_point.point_data[0]);

  const std::size_t offset = chunk_id * args->chunk_size;
  if (offset >= args->items) return;

  const std::size_t count = std::min(args->chunk_size, args->items - offset);
  const std::size_t start_idx = args->level_start - offset;

  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  sift_down_range_raw(acc, args->n, start_idx, count);
}

void write_heap_characteristics(const std::vector<int>& v) {
  std::ofstream outFile("heaps.txt");

  std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i) {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  std::size_t last_start = (v.size() > print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = last_start; i < v.size(); ++i) {
    outFile << v[i];
    if (i + 1 < v.size()) outFile << " ";
  }

  outFile << "\n";
  long long sum = std::accumulate(v.begin(), v.end(), 0LL);
  outFile << "Sum of all elements: " << sum << "\n";

  if (!v.empty()) {
    outFile << "Root (max) element: " << v[0] << "\n";
  }

  bool is_valid_heap = std::is_heap(v.begin(), v.end());
  outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";
  outFile.close();
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
  Options opts = parse_options(Runtime::get_input_args());
  const std::size_t vector_size = opts.vector_size;

  if (vector_size == 0) {
    write_heap_characteristics(std::vector<int>{});
    return;
  }

  std::size_t chunk_size = opts.chunk_size;
  if (chunk_size == 0) {
    std::size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    chunk_size = vector_size / threads;
    if (chunk_size == 0) chunk_size = 1;
  }

  // Create Legion region to store vector<int>
  Rect<1> elem_rect(Point<1>(0), Point<1>(static_cast<coord_t>(vector_size - 1)));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Initialize with iota: 0,1,2,...,vector_size-1
  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
    for (std::size_t i = 0; i < vector_size; ++i) {
      acc[Point<1>(static_cast<coord_t>(i))] = static_cast<int>(i);
    }

    runtime->unmap_region(ctx, pr);
  }

  // Bottom-up, level-parallel heap construction
  if (vector_size > 1) {
    std::size_t start = (vector_size - 2) / 2;

    while (start > 0) {
      const std::size_t end_level = rightmost_parent_of_prev_level(start);
      const std::size_t items = start - end_level; // parent indices: start .. end_level+1

      std::size_t effective_chunk = chunk_size;
      if (effective_chunk > items) {
        effective_chunk = std::max<std::size_t>(1, items / 2);
      }
      if (effective_chunk == 0) effective_chunk = 1;

      const std::size_t num_chunks = (items + effective_chunk - 1) / effective_chunk;

      LevelTaskArgs level_args;
      level_args.n = vector_size;
      level_args.level_start = start;
      level_args.chunk_size = effective_chunk;
      level_args.items = items;

      Rect<1> launch_rect(Point<1>(0), Point<1>(static_cast<coord_t>(num_chunks - 1)));
      IndexLauncher launcher(
          SIFT_LEVEL_TASK_ID,
          Domain(launch_rect),
          TaskArgument(&level_args, sizeof(level_args)),
          ArgumentMap());

      // SIMULTANEOUS coherence allows these level tasks to run concurrently
      // (the algorithm ensures they touch disjoint subtrees within a level).
      launcher.add_region_requirement(RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
      launcher.region_requirements[0].add_field(FID_VALUE);

      FutureMap fm = runtime->execute_index_space(ctx, launcher);
      fm.wait_all_results(); // level barrier

      start = end_level;
    }

    // Sift root (index 0)
    LevelTaskArgs root_args;
    root_args.n = vector_size;
    root_args.level_start = 0;
    root_args.chunk_size = 1;
    root_args.items = 1;

    Rect<1> root_rect(Point<1>(0), Point<1>(0));
    IndexLauncher root_launcher(
        SIFT_LEVEL_TASK_ID,
        Domain(root_rect),
        TaskArgument(&root_args, sizeof(root_args)),
        ArgumentMap());

    root_launcher.add_region_requirement(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    root_launcher.region_requirements[0].add_field(FID_VALUE);

    FutureMap root_fm = runtime->execute_index_space(ctx, root_launcher);
    root_fm.wait_all_results();
  }

  // Copy back to std::vector for output checks
  std::vector<int> v(vector_size);
  {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
    for (std::size_t i = 0; i < vector_size; ++i) {
      v[i] = acc[Point<1>(static_cast<coord_t>(i))];
    }

    runtime->unmap_region(ctx, pr);
  }

  write_heap_characteristics(v);

  // Cleanup Legion objects
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(SIFT_LEVEL_TASK_ID, "sift_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_level_task>(registrar, "sift_level_task");
  }

  return Runtime::start(argc, argv);
}
