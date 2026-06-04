// Translated from HPX to Legion execution model (default mapper, no custom mapping).
#include "legion.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;
namespace po = boost::program_options;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID = 2
};

struct SiftRangeArgs {
  std::vector<int>* vec;
  std::ptrdiff_t len;
  std::size_t start_index;
  std::size_t count;
};

// Integer helper equivalent to: 2^(floor(log2(start))) - 2
static inline std::ptrdiff_t next_level_rightmost(std::ptrdiff_t start) {
  if (start <= 0) return -1;
  std::ptrdiff_t p = 1;
  while ((p << 1) > 0 && (p << 1) <= start) {
    p <<= 1;
  }
  return p - 2;
}

void sift_range_task(const Task* task,
                     const std::vector<PhysicalRegion>& /*regions*/,
                     Context /*ctx*/,
                     Runtime* /*runtime*/) {
  const auto* args = static_cast<const SiftRangeArgs*>(task->args);
  if (args == nullptr || args->vec == nullptr || args->count == 0) return;

  std::vector<int>& v = *(args->vec);
  if (v.empty()) return;
  if (args->start_index >= v.size()) return;

  sift_down_range(v.begin(), v.end(), std::less<int>(), args->len,
                  v.begin() + static_cast<std::ptrdiff_t>(args->start_index),
                  args->count);
}

void _make_heap(Runtime* runtime, Context ctx, std::vector<int>& v, std::size_t chunk_size) {
  using difference_type = std::ptrdiff_t;
  const difference_type n = static_cast<difference_type>(v.size());
  if (n <= 1) return;

  if (chunk_size == 0) chunk_size = 1;

  // Bottom-up, level-parallel heapify.
  for (difference_type start = (n - 2) / 2; start > 0; start = next_level_rightmost(start)) {
    const difference_type end_level = next_level_rightmost(start);
    const std::size_t items = static_cast<std::size_t>(start - end_level);

    if (items == 0) continue;

    std::size_t effective_chunk = chunk_size;
    if (effective_chunk > items) {
      // Keep behavior close to original intent, but avoid zero chunk.
      effective_chunk = std::max<std::size_t>(1, items / 2);
    }
    effective_chunk = std::max<std::size_t>(1, effective_chunk);

    std::vector<Future> workitems;
    workitems.reserve((items + effective_chunk - 1) / effective_chunk);

    std::size_t cnt = 0;
    while (cnt + effective_chunk < items) {
      SiftRangeArgs args{
          &v,
          n,
          static_cast<std::size_t>(start) - cnt,
          effective_chunk
      };
      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      workitems.push_back(runtime->execute_task(ctx, launcher));
      cnt += effective_chunk;
    }

    if (cnt < items) {
      SiftRangeArgs args{
          &v,
          n,
          static_cast<std::size_t>(start) - cnt,
          items - cnt
      };
      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      workitems.push_back(runtime->execute_task(ctx, launcher));
    }

    // Synchronize per level.
    for (auto& f : workitems) {
      f.get_void_result();
    }
  }

  // Final sift-down for root.
  sift_down(v.begin(), v.end(), std::less<int>(), n, v.begin());
}

void write_heap_characteristics(const std::vector<int>& v) {
  std::ofstream outFile("heaps.txt");
  if (!outFile.is_open()) {
    std::cerr << "Failed to open heaps.txt for writing\n";
    return;
  }

  const std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i) {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  const std::size_t start_last = (v.size() >= print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = start_last; i < v.size(); ++i) {
    outFile << v[i];
    if (i + 1 < v.size()) outFile << " ";
  }

  outFile << "\n";
  const long long sum = std::accumulate(v.begin(), v.end(), 0LL);
  outFile << "Sum of all elements: " << sum << "\n";

  if (!v.empty()) {
    outFile << "Root (max) element: " << v[0] << "\n";
  }

  const bool is_valid_heap = std::is_heap(v.begin(), v.end());
  outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;

  try {
    InputArgs args = Runtime::get_input_args();

    po::options_description cmdline("usage: heaps [options]");
    cmdline.add_options()
      ("vector_size", po::value<std::size_t>()->default_value(25), "size of vector")
      ("chunk_size", po::value<std::size_t>()->default_value(0), "amount of work per task");

    po::variables_map vm;
    auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(cmdline)
                    .allow_unregistered() // ignore Legion runtime flags
                    .run();
    po::store(parsed, vm);
    po::notify(vm);

    vector_size = vm["vector_size"].as<std::size_t>();
    chunk_size = vm["chunk_size"].as<std::size_t>();
  } catch (const std::exception& e) {
    std::cerr << "Command line parse error: " << e.what() << "\n";
    return;
  }

  std::vector<int> v(vector_size);
  std::iota(v.begin(), v.end(), 0);

  if (chunk_size == 0) {
    std::size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    chunk_size = std::max<std::size_t>(1, vector_size / threads);
  }

  _make_heap(runtime, ctx, v, chunk_size);
  write_heap_characteristics(v);
}

int main(int argc, char* argv[]) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(SIFT_RANGE_TASK_ID, "sift_range");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_range_task>(registrar, "sift_range");
  }

  return Runtime::start(argc, argv);
}
