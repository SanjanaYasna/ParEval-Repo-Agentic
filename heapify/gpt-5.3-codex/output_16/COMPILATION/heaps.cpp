#include "legion.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_CHUNK_TASK_ID = 2
};

enum FieldIDs {
  FID_VALUE = 1
};

struct LevelTaskArgs {
  int64_t n;
  int64_t level_start;  // rightmost node index of the current level
  int64_t items;        // number of nodes in this level
  int64_t chunk_size;   // nodes per task
};

template <typename Accessor>
static inline void sift_down_accessor(Accessor &acc, int64_t len, int64_t start_idx) {
  int64_t child = start_idx;

  if (len < 2 || ((len - 2) / 2) < child) return;

  child = 2 * child + 1;

  if ((child + 1) < len &&
      acc[Point<1>(static_cast<coord_t>(child))] <
          acc[Point<1>(static_cast<coord_t>(child + 1))]) {
    ++child;
  }

  if (acc[Point<1>(static_cast<coord_t>(child))] <
      acc[Point<1>(static_cast<coord_t>(start_idx))]) {
    return;
  }

  int top = acc[Point<1>(static_cast<coord_t>(start_idx))];
  int64_t hole = start_idx;

  do {
    acc[Point<1>(static_cast<coord_t>(hole))] =
        acc[Point<1>(static_cast<coord_t>(child))];
    hole = child;

    if (((len - 2) / 2) < child) break;

    child = 2 * child + 1;

    if ((child + 1) < len &&
        acc[Point<1>(static_cast<coord_t>(child))] <
            acc[Point<1>(static_cast<coord_t>(child + 1))]) {
      ++child;
    }
  } while (!(acc[Point<1>(static_cast<coord_t>(child))] < top));

  acc[Point<1>(static_cast<coord_t>(hole))] = top;
}

static inline int64_t floor_log2_u64(uint64_t x) {
  int64_t l = -1;
  while (x) {
    x >>= 1;
    ++l;
  }
  return l;
}

static void write_heap_characteristics(const std::vector<int> &v) {
  std::ofstream outFile("heaps.txt");

  const std::size_t print_count = std::min<std::size_t>(10, v.size());

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i) {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  const std::size_t start_last = (v.size() > print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = start_last; i < v.size(); ++i) {
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
}

void sift_chunk_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context /*ctx*/,
                     Runtime * /*runtime*/) {
  const auto *args = static_cast<const LevelTaskArgs *>(task->args);
  const int64_t task_idx = static_cast<int64_t>(task->index_point[0]);

  const int64_t offset = task_idx * args->chunk_size;
  if (offset >= args->items) return;

  const int64_t count = std::min<int64_t>(args->chunk_size, args->items - offset);
  const int64_t start_idx = args->level_start - offset;

  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);

  for (int64_t i = 0; i < count; ++i) {
    sift_down_accessor(acc, args->n, start_idx - i);
  }
}

static void parse_command_line(const InputArgs &in_args,
                               std::size_t &vector_size,
                               std::size_t &chunk_size,
                               bool &help) {
  vector_size = 25;
  chunk_size = 0;
  help = false;

  for (int i = 1; i < in_args.argc; ++i) {
    std::string a(in_args.argv[i]);

    if (a == "--help" || a == "-h") {
      help = true;
      continue;
    }

    if (a == "--vector_size" && i + 1 < in_args.argc) {
      vector_size = static_cast<std::size_t>(std::stoull(in_args.argv[++i]));
      continue;
    }
    if (a.rfind("--vector_size=", 0) == 0) {
      vector_size = static_cast<std::size_t>(std::stoull(a.substr(14)));
      continue;
    }

    if (a == "--chunk_size" && i + 1 < in_args.argc) {
      chunk_size = static_cast<std::size_t>(std::stoull(in_args.argv[++i]));
      continue;
    }
    if (a.rfind("--chunk_size=", 0) == 0) {
      chunk_size = static_cast<std::size_t>(std::stoull(a.substr(13)));
      continue;
    }
  }
}

void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx,
                    Runtime *runtime) {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
  bool help = false;

  parse_command_line(Runtime::get_input_args(), vector_size, chunk_size, help);

  if (help) {
    std::cout << "Usage: ./heaps [--vector_size N] [--chunk_size C] [Legion runtime options]\n";
    std::cout << "Example: ./heaps --vector_size 999 --chunk_size 100\n";
    return;
  }

  if (chunk_size == 0) {
    unsigned threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    chunk_size = vector_size / threads;
    if (chunk_size == 0) chunk_size = 1;
  }

  if (vector_size == 0) {
    write_heap_characteristics(std::vector<int>{});
    return;
  }

  // Create 1D region storing the array.
  Rect<1> elem_rect(Point<1>(0), Point<1>(static_cast<coord_t>(vector_size - 1)));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Initialize v[i] = i.
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

  // Bottom-up, level-parallel heap construction.
  const int64_t n = static_cast<int64_t>(vector_size);
  if (n > 1) {
    const int64_t max_parent = (n - 2) / 2;
    const int64_t deepest_internal_level = floor_log2_u64(static_cast<uint64_t>(max_parent + 1));

    for (int64_t level = deepest_internal_level; level >= 1; --level) {
      const int64_t level_left = (1LL << level) - 1;
      const int64_t level_right = std::min<int64_t>((1LL << (level + 1)) - 2, max_parent);
      const int64_t items = level_right - level_left + 1;
      if (items <= 0) continue;

      int64_t effective_chunk = static_cast<int64_t>(chunk_size);
      if (effective_chunk <= 0) effective_chunk = 1;
      if (effective_chunk > items) effective_chunk = items;

      const int64_t num_tasks = (items + effective_chunk - 1) / effective_chunk;

      LevelTaskArgs args{n, level_right, items, effective_chunk};

      Domain launch_domain(Rect<1>(
          Point<1>(0), Point<1>(static_cast<coord_t>(num_tasks - 1))));
      ArgumentMap arg_map;

      IndexLauncher launcher(
          SIFT_CHUNK_TASK_ID, launch_domain, TaskArgument(&args, sizeof(args)), arg_map);

      RegionRequirement rr(lr, READ_WRITE, SIMULTANEOUS, lr);
      rr.add_field(FID_VALUE);
      launcher.add_region_requirement(rr);

      FutureMap fm = runtime->execute_index_space(ctx, launcher);
      fm.wait_all_results();
    }

    // Root sift-down
    RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
    sift_down_accessor(acc, n, 0);

    // Snapshot for output checks.
    std::vector<int> v(vector_size);
    for (std::size_t i = 0; i < vector_size; ++i) {
      v[i] = acc[Point<1>(static_cast<coord_t>(i))];
    }

    runtime->unmap_region(ctx, pr);

    write_heap_characteristics(v);
  } else {
    // n == 1
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
    std::vector<int> v(1);
    v[0] = acc[Point<1>(0)];

    runtime->unmap_region(ctx, pr);
    write_heap_characteristics(v);
  }

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(SIFT_CHUNK_TASK_ID, "sift_chunk_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_chunk_task>(registrar, "sift_chunk_task");
  }

  return Runtime::start(argc, argv);
}
