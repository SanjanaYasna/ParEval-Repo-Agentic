#include "legion.h"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs
{
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID
};

enum FieldIDs
{
  FID_VAL = 1
};

struct SiftRangeArgs
{
  std::int64_t len;
  std::int64_t start_idx;
  std::size_t count;
};

struct ProgramOptions
{
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
};

static bool parse_size_t(const std::string &s, std::size_t &out)
{
  try
  {
    std::size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos);
    if (pos != s.size()) return false;
    out = static_cast<std::size_t>(v);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

static ProgramOptions parse_options(int argc, char **argv)
{
  ProgramOptions opts;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg(argv[i]);

    if (arg == "--vector_size" && i + 1 < argc)
    {
      std::size_t parsed = 0;
      if (parse_size_t(argv[i + 1], parsed)) opts.vector_size = parsed;
      ++i;
    }
    else if (arg.rfind("--vector_size=", 0) == 0)
    {
      std::size_t parsed = 0;
      if (parse_size_t(arg.substr(std::string("--vector_size=").size()), parsed))
        opts.vector_size = parsed;
    }
    else if (arg == "--chunk_size" && i + 1 < argc)
    {
      std::size_t parsed = 0;
      if (parse_size_t(argv[i + 1], parsed)) opts.chunk_size = parsed;
      ++i;
    }
    else if (arg.rfind("--chunk_size=", 0) == 0)
    {
      std::size_t parsed = 0;
      if (parse_size_t(arg.substr(std::string("--chunk_size=").size()), parsed))
        opts.chunk_size = parsed;
    }
  }

  return opts;
}

static std::size_t get_available_cpu_processors()
{
  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);

  std::size_t count = 0;
  for (auto it = pq.begin(); it != pq.end(); ++it)
    ++count;

  return std::max<std::size_t>(1, count);
}

static std::ptrdiff_t highest_power_of_two_leq(std::ptrdiff_t x)
{
  // x > 0 expected
  std::uint64_t v = static_cast<std::uint64_t>(x);
  std::uint64_t p = 1;
  while ((p << 1) <= v)
    p <<= 1;
  return static_cast<std::ptrdiff_t>(p);
}

void write_heap_characteristics(const std::vector<int> &v)
{
  std::ofstream outFile("heaps.txt");

  const std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i)
  {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  if (!v.empty())
  {
    std::size_t start_last = (v.size() > print_count) ? (v.size() - print_count) : 0;
    for (std::size_t i = start_last; i < v.size(); ++i)
    {
      outFile << v[i];
      if (i + 1 < v.size()) outFile << " ";
    }
  }

  outFile << "\n";
  long long sum = std::accumulate(v.begin(), v.end(), 0LL);
  outFile << "Sum of all elements: " << sum << "\n";

  if (!v.empty())
    outFile << "Root (max) element: " << v[0] << "\n";

  bool is_valid_heap = std::is_heap(v.begin(), v.end());
  outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";
}

void sift_range_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx,
                     Runtime *runtime)
{
  const SiftRangeArgs *args = static_cast<const SiftRangeArgs *>(task->args);
  if (args == nullptr || args->len <= 1 || args->count == 0) return;

  const std::ptrdiff_t len = static_cast<std::ptrdiff_t>(args->len);
  const std::ptrdiff_t start_idx = static_cast<std::ptrdiff_t>(args->start_idx);
  if (start_idx < 0 || start_idx >= len) return;

  using RWAccessor =
      FieldAccessor<READ_WRITE, int, 1, coord_t, Realm::AffineAccessor<int, 1, coord_t>>;

  RWAccessor acc(regions[0], FID_VAL);

  Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> rect = domain.bounds<1, coord_t>();

  int *base = acc.ptr(rect.lo);

  std::size_t safe_count = std::min<std::size_t>(args->count, static_cast<std::size_t>(start_idx + 1));
  sift_down_range<int *, std::less<int>>(
      base, base + len, std::less<int>{}, len, base + start_idx, safe_count);
}

static void make_heap_legion(Runtime *runtime,
                             Context ctx,
                             LogicalRegion lr,
                             std::size_t n,
                             std::size_t chunk_size)
{
  if (n <= 1) return;

  const std::ptrdiff_t len = static_cast<std::ptrdiff_t>(n);

  for (std::ptrdiff_t level_last = (len - 2) / 2; level_last > 0;)
  {
    const std::ptrdiff_t level_first = highest_power_of_two_leq(level_last + 1) - 1;
    const std::size_t items = static_cast<std::size_t>(level_last - level_first + 1);

    std::size_t level_chunk = chunk_size;
    if (level_chunk == 0 || level_chunk > items) level_chunk = items;
    if (level_chunk == 0) level_chunk = 1;

    std::vector<Future> futures;
    futures.reserve((items + level_chunk - 1) / level_chunk);

    std::size_t cnt = 0;
    while (cnt < items)
    {
      std::size_t this_count = std::min(level_chunk, items - cnt);
      SiftRangeArgs args{
          static_cast<std::int64_t>(len),
          static_cast<std::int64_t>(level_last - static_cast<std::ptrdiff_t>(cnt)),
          this_count};

      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      RegionRequirement rr(lr, READ_WRITE, SIMULTANEOUS, lr);
      rr.add_field(FID_VAL);
      launcher.add_region_requirement(rr);

      futures.emplace_back(runtime->execute_task(ctx, launcher));
      cnt += this_count;
    }

    for (auto &f : futures)
      f.get_void_result();

    level_last = level_first - 1;
  }

  // Sift down the root node
  SiftRangeArgs root_args{
      static_cast<std::int64_t>(len),
      0,
      1};

  TaskLauncher root_launcher(SIFT_RANGE_TASK_ID, TaskArgument(&root_args, sizeof(root_args)));
  RegionRequirement root_rr(lr, READ_WRITE, EXCLUSIVE, lr);
  root_rr.add_field(FID_VAL);
  root_launcher.add_region_requirement(root_rr);
  runtime->execute_task(ctx, root_launcher).get_void_result();
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx,
                    Runtime *runtime)
{
  const InputArgs &in = Runtime::get_input_args();
  ProgramOptions opts = parse_options(in.argc, in.argv);

  std::size_t vector_size = opts.vector_size;
  std::size_t chunk_size = opts.chunk_size;

  if (chunk_size == 0)
  {
    std::size_t threads = get_available_cpu_processors();
    chunk_size = (threads > 0) ? (vector_size / threads) : vector_size;
    if (chunk_size == 0) chunk_size = 1;
  }

  if (vector_size == 0)
  {
    write_heap_characteristics(std::vector<int>{});
    return;
  }

  Rect<1> elem_rect(0, static_cast<coord_t>(vector_size - 1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VAL);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Initialize v[i] = i
  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    using WDAccessor =
        FieldAccessor<WRITE_DISCARD, int, 1, coord_t, Realm::AffineAccessor<int, 1, coord_t>>;
    WDAccessor acc(pr, FID_VAL);

    Domain domain = runtime->get_index_space_domain(ctx, is);
    Rect<1> rect = domain.bounds<1, coord_t>();
    int *base = acc.ptr(rect.lo);

    for (std::size_t i = 0; i < vector_size; ++i)
      base[i] = static_cast<int>(i);

    runtime->unmap_region(ctx, pr);
  }

  make_heap_legion(runtime, ctx, lr, vector_size, chunk_size);

  // Copy result back to std::vector for output and validation
  std::vector<int> v(vector_size);
  {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    using ROAccessor =
        FieldAccessor<READ_ONLY, int, 1, coord_t, Realm::AffineAccessor<int, 1, coord_t>>;
    ROAccessor acc(pr, FID_VAL);

    Domain domain = runtime->get_index_space_domain(ctx, is);
    Rect<1> rect = domain.bounds<1, coord_t>();
    const int *base = acc.ptr(rect.lo);

    std::copy(base, base + vector_size, v.begin());
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
