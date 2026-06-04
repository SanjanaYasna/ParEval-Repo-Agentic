#include "legion.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp" // kept for repository consistency

using namespace Legion;

enum TaskIDs
{
  TOP_LEVEL_TASK_ID = 1,
  SIFT_CHUNK_TASK_ID
};

enum FieldIDs
{
  FID_VALUE = 1
};

struct Config
{
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
};

struct SiftChunkArgs
{
  std::int64_t len;
  std::int64_t start;
  std::int64_t count;
};

static inline Point<1> p1(std::int64_t i)
{
  return Point<1>(static_cast<coord_t>(i));
}

template <typename Accessor, typename Pred>
void sift_down_accessor(Accessor &acc, Pred pred, std::int64_t len, std::int64_t start)
{
  std::int64_t child = start;

  if (len < 2 || ((len - 2) / 2) < child)
    return;

  child = 2 * child + 1;

  if ((child + 1) < len && pred(acc[p1(child)], acc[p1(child + 1)]))
    ++child;

  if (pred(acc[p1(child)], acc[p1(start)]))
    return;

  int top = acc[p1(start)];
  std::int64_t root = start;

  do
  {
    acc[p1(root)] = acc[p1(child)];
    root = child;

    if (((len - 2) / 2) < child)
      break;

    child = 2 * child + 1;
    if ((child + 1) < len && pred(acc[p1(child)], acc[p1(child + 1)]))
      ++child;

  } while (!pred(acc[p1(child)], top));

  acc[p1(root)] = top;
}

void sift_chunk_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx,
                     Runtime *runtime)
{
  (void)ctx;
  (void)runtime;

  if (task == nullptr || task->arglen != sizeof(SiftChunkArgs) || regions.size() != 1)
    return;

  const SiftChunkArgs &args = *reinterpret_cast<const SiftChunkArgs *>(task->args);
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);

  const std::int64_t n = args.len;
  for (std::int64_t i = 0; i < args.count; ++i)
  {
    const std::int64_t idx = args.start - i;
    if (idx < 0 || idx >= n)
      continue;
    sift_down_accessor(acc, std::less<int>(), n, idx);
  }
}

void make_heap_legion(Runtime *runtime, Context ctx, LogicalRegion lr, std::size_t n, std::size_t chunk_size)
{
  (void)chunk_size;

  if (n <= 1)
    return;

  RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
  req.add_field(FID_VALUE);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
  const std::int64_t len = static_cast<std::int64_t>(n);

  for (std::int64_t start = (len - 2) / 2; start >= 0; --start)
  {
    sift_down_accessor(acc, std::less<int>(), len, start);
    if (start == 0)
      break;
  }

  runtime->unmap_region(ctx, pr);
}

void write_heap_characteristics(const std::vector<int> &v)
{
  std::ofstream outFile("heaps.txt");

  const std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i)
  {
    outFile << v[i];
    if (i + 1 < print_count)
      outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  const std::size_t start_last = (v.size() > print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = start_last; i < v.size(); ++i)
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

Config parse_config(const InputArgs &args)
{
  Config cfg;

  for (int i = 1; i < args.argc; ++i)
  {
    std::string a(args.argv[i]);

    auto parse_opt = [&](const char *name, std::size_t &out) -> bool {
      const std::string key(name);

      if (a == key)
      {
        if (i + 1 < args.argc)
        {
          try
          {
            out = static_cast<std::size_t>(std::stoull(args.argv[++i]));
          }
          catch (...)
          {
            // keep default if parsing fails
          }
        }
        return true;
      }

      const std::string eq = key + "=";
      if (a.rfind(eq, 0) == 0)
      {
        try
        {
          out = static_cast<std::size_t>(std::stoull(a.substr(eq.size())));
        }
        catch (...)
        {
          // keep default if parsing fails
        }
        return true;
      }

      return false;
    };

    if (parse_opt("--vector_size", cfg.vector_size))
      continue;
    if (parse_opt("--chunk_size", cfg.chunk_size))
      continue;
  }

  return cfg;
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime)
{
  (void)task;
  (void)regions;

  const Config cfg = parse_config(Runtime::get_input_args());
  const std::size_t vector_size = cfg.vector_size;

  std::size_t chunk_size = cfg.chunk_size;
  if (chunk_size == 0)
  {
    const unsigned hw = std::thread::hardware_concurrency();
    const std::size_t threads = (hw == 0 ? 1 : static_cast<std::size_t>(hw));
    chunk_size = std::max<std::size_t>(1, (vector_size == 0 ? 1 : vector_size / threads));
  }

  if (vector_size == 0)
  {
    write_heap_characteristics(std::vector<int>{});
    return;
  }

  // Create Legion region backing the heap array
  Rect<1> elem_rect(p1(0), p1(static_cast<std::int64_t>(vector_size - 1)));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Initialize array: 0,1,2,...,vector_size-1
  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
    for (std::size_t i = 0; i < vector_size; ++i)
      acc[p1(static_cast<std::int64_t>(i))] = static_cast<int>(i);

    runtime->unmap_region(ctx, pr);
  }

  // Heapify
  make_heap_legion(runtime, ctx, lr, vector_size, chunk_size);

  // Read back and write diagnostics
  std::vector<int> v(vector_size);
  {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
    for (std::size_t i = 0; i < vector_size; ++i)
      v[i] = acc[p1(static_cast<std::int64_t>(i))];

    runtime->unmap_region(ctx, pr);
  }

  write_heap_characteristics(v);

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(SIFT_CHUNK_TASK_ID, "sift_chunk");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<sift_chunk_task>(registrar, "sift_chunk");
  }

  return Runtime::start(argc, argv);
}
