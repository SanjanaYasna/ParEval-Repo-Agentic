#include "legion.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using namespace Legion;

enum TaskIDs
{
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID
};

enum FieldIDs
{
  FID_VALUE = 1
};

struct SiftRangeArgs
{
  int64_t start;
  int64_t count;
  int64_t len;
};

struct AppOptions
{
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
  bool show_help = false;
};

inline Point<1> p1(int64_t i)
{
  return Point<1>(static_cast<coord_t>(i));
}

template <typename Accessor>
inline int acc_read(const Accessor &acc, int64_t idx)
{
  return acc[p1(idx)];
}

template <typename Accessor>
inline void acc_write(Accessor &acc, int64_t idx, int value)
{
  acc[p1(idx)] = value;
}

// Equivalent of sift_down, but operating on a Legion accessor
template <typename Accessor, typename Pred>
void sift_down_accessor(Accessor &acc, const Pred &pred, int64_t len, int64_t start)
{
  int64_t child = start;

  if (len < 2 || (len - 2) / 2 < child)
    return;

  child = 2 * child + 1;

  int child_val = acc_read(acc, child);
  if ((child + 1) < len)
  {
    int right_val = acc_read(acc, child + 1);
    if (pred(child_val, right_val))
    {
      ++child;
      child_val = right_val;
    }
  }

  int start_val = acc_read(acc, start);
  if (pred(child_val, start_val))
    return;

  int top = start_val;
  do
  {
    acc_write(acc, start, child_val);
    start = child;

    if ((len - 2) / 2 < child)
      break;

    child = 2 * child + 1;
    child_val = acc_read(acc, child);

    if ((child + 1) < len)
    {
      int right_val = acc_read(acc, child + 1);
      if (pred(child_val, right_val))
      {
        ++child;
        child_val = right_val;
      }
    }
  } while (!pred(child_val, top));

  acc_write(acc, start, top);
}

template <typename Accessor, typename Pred>
void sift_down_range_accessor(Accessor &acc, const Pred &pred, int64_t len, int64_t start, int64_t count)
{
  for (int64_t i = 0; i < count; ++i)
  {
    sift_down_accessor(acc, pred, len, start - i);
  }
}

// Returns index immediately before the start of `start`'s heap level.
// For 0-based heap indexing, level starts are 2^L - 1.
inline int64_t level_boundary(int64_t start)
{
  if (start <= 0)
    return 0;

  uint64_t x = static_cast<uint64_t>(start) + 1; // 1-based index
  uint64_t p = 1;
  while (p <= (x >> 1))
    p <<= 1; // largest power-of-two <= x

  return static_cast<int64_t>(p) - 2;
}

void sift_range_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context /*ctx*/, Runtime * /*runtime*/)
{
  assert(task != nullptr);
  assert(task->arglen == sizeof(SiftRangeArgs));
  assert(regions.size() == 1);

  const SiftRangeArgs *args = static_cast<const SiftRangeArgs *>(task->args);
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);

  sift_down_range_accessor(acc, std::less<int>(), args->len, args->start, args->count);
}

void make_heap_legion(Runtime *runtime, Context ctx, LogicalRegion heap_lr, int64_t n, std::size_t chunk_size)
{
  if (n <= 1)
    return;

  for (int64_t start = (n - 2) / 2; start > 0;)
  {
    int64_t end_level = level_boundary(start);
    std::size_t items = static_cast<std::size_t>(start - end_level);

    std::size_t effective_chunk = (chunk_size == 0) ? items : std::min(chunk_size, items);
    if (effective_chunk == 0)
      effective_chunk = 1;

    std::vector<Future> futures;
    futures.reserve((items + effective_chunk - 1) / effective_chunk);

    std::size_t cnt = 0;
    while (cnt < items)
    {
      const std::size_t work = std::min(effective_chunk, items - cnt);
      SiftRangeArgs args{
          start - static_cast<int64_t>(cnt),
          static_cast<int64_t>(work),
          n};

      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      launcher.add_region_requirement(RegionRequirement(heap_lr, READ_WRITE, SIMULTANEOUS, heap_lr));
      launcher.add_field(0, FID_VALUE);
      futures.push_back(runtime->execute_task(ctx, launcher));

      cnt += work;
    }

    // Per-level synchronization barrier
    for (Future &f : futures)
      f.get_void_result();

    start = end_level;
  }

  // Root node
  SiftRangeArgs root_args{0, 1, n};
  TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&root_args, sizeof(root_args)));
  launcher.add_region_requirement(RegionRequirement(heap_lr, READ_WRITE, EXCLUSIVE, heap_lr));
  launcher.add_field(0, FID_VALUE);
  runtime->execute_task(ctx, launcher).get_void_result();
}

void write_heap_characteristics(const std::vector<int> &v)
{
  std::ofstream outFile("heaps.txt");

  std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i)
  {
    outFile << v[i];
    if (i + 1 < print_count)
      outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  std::size_t start_idx = (v.size() >= print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = start_idx; i < v.size(); ++i)
  {
    outFile << v[i];
    if (i + 1 < v.size())
      outFile << " ";
  }

  outFile << "\n";
  long long sum = std::accumulate(v.begin(), v.end(), 0LL);
  outFile << "Sum of all elements: " << sum << "\n";

  if (!v.empty())
    outFile << "Root (max) element: " << v[0] << "\n";

  bool is_valid_heap = std::is_heap(v.begin(), v.end());
  outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";
}

AppOptions parse_options(const InputArgs &input_args)
{
  namespace po = boost::program_options;

  AppOptions opts;
  po::options_description cmdline("usage: heaps [options]");
  cmdline.add_options()
      ("help,h", "show help")
      ("vector_size", po::value<std::size_t>()->default_value(25), "size of vector")
      ("chunk_size", po::value<std::size_t>()->default_value(0), "amount of work per task");

  po::variables_map vm;
  auto parsed =
      po::command_line_parser(input_args.argc, input_args.argv)
          .options(cmdline)
          .allow_unregistered() // allow Legion runtime flags
          .run();

  po::store(parsed, vm);
  po::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << cmdline << std::endl;
    opts.show_help = true;
    return opts;
  }

  opts.vector_size = vm["vector_size"].as<std::size_t>();
  opts.chunk_size = vm["chunk_size"].as<std::size_t>();
  return opts;
}

void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
  AppOptions opts = parse_options(Runtime::get_input_args());
  if (opts.show_help)
    return;

  std::size_t vector_size = opts.vector_size;
  std::size_t chunk_size = opts.chunk_size;

  if (chunk_size == 0)
  {
    std::size_t threads = std::max<unsigned>(1u, std::thread::hardware_concurrency());
    chunk_size = vector_size / threads;
    if (chunk_size == 0)
      chunk_size = 1;
  }

  std::vector<int> v(vector_size);

  if (vector_size > 0)
  {
    Rect<1> elem_rect(Point<1>(0), Point<1>(static_cast<coord_t>(vector_size - 1)));
    IndexSpace is = runtime->create_index_space(ctx, elem_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(int), FID_VALUE);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Initialize heap array with 0..vector_size-1
    {
      RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      pr.wait_until_valid();

      FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
      for (std::size_t i = 0; i < vector_size; ++i)
        acc[p1(static_cast<int64_t>(i))] = static_cast<int>(i);

      runtime->unmap_region(ctx, pr);
    }

    make_heap_legion(runtime, ctx, lr, static_cast<int64_t>(vector_size), chunk_size);

    // Read back result
    {
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      pr.wait_until_valid();

      FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
      for (std::size_t i = 0; i < vector_size; ++i)
        v[i] = acc[p1(static_cast<int64_t>(i))];

      runtime->unmap_region(ctx, pr);
    }

    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
  }

  write_heap_characteristics(v);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(SIFT_RANGE_TASK_ID, "sift_range_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_range_task>(registrar, "sift_range_task");
  }

  return Runtime::start(argc, argv);
}
