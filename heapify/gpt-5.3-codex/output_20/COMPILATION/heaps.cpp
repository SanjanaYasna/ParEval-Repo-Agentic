// Translated from HPX to Legion execution model (default mapper assumptions)
// source inspiration: https://github.com/Syntaf/heapify/tree/master

#include "legion.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID
};

struct SiftRangeArgs {
  int* data;               // shared heap array pointer (single-node assumption)
  std::size_t n;           // total length
  std::size_t start_index; // starting heap node index
  std::size_t count;       // number of nodes to process descending from start_index
};

void sift_range_task(const Task* task,
                     const std::vector<PhysicalRegion>& /*regions*/,
                     Context /*ctx*/,
                     Runtime* /*runtime*/) {
  const auto* args = static_cast<const SiftRangeArgs*>(task->args);
  int* first = args->data;
  int* last = args->data + args->n;

  sift_down_range<int*, std::less<int>>(
      first, last, std::less<int>(), static_cast<std::ptrdiff_t>(args->n),
      first + static_cast<std::ptrdiff_t>(args->start_index), args->count);
}

static void make_heap_legion(std::vector<int>& v, std::size_t chunk_size,
                             Context ctx, Runtime* runtime) {
  using difference_type = std::ptrdiff_t;
  const difference_type n = static_cast<difference_type>(v.size());
  if (n <= 1) return;

  if (chunk_size == 0) chunk_size = 1;
  int* data = v.data();

  const difference_type last_parent = (n - 2) / 2;

  // Build parent-index level ranges [begin, end]
  std::vector<std::pair<difference_type, difference_type>> levels;
  for (difference_type begin = 0, end = 0; begin <= last_parent;) {
    levels.emplace_back(begin, std::min(end, last_parent));

    const difference_type maxv = std::numeric_limits<difference_type>::max();
    if (begin > (maxv - 1) / 2 || end > (maxv - 2) / 2) break;
    begin = 2 * begin + 1;
    end = 2 * end + 2;
  }

  // Process bottom-up by level, excluding root (handled once at the end)
  for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
    const difference_type level_begin = it->first;
    const difference_type level_end = it->second;
    if (level_begin == 0) continue;  // root level

    const std::size_t items =
        static_cast<std::size_t>(level_end - level_begin + 1);
    if (items == 0) continue;

    const std::size_t level_chunk =
        std::max<std::size_t>(1, std::min<std::size_t>(chunk_size, items));

    std::vector<Future> futures;
    futures.reserve((items + level_chunk - 1) / level_chunk);

    std::size_t processed = 0;
    while (processed < items) {
      const std::size_t cnt = std::min(level_chunk, items - processed);
      SiftRangeArgs args{
          data,
          static_cast<std::size_t>(n),
          static_cast<std::size_t>(
              level_end - static_cast<difference_type>(processed)),
          cnt};

      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      futures.emplace_back(runtime->execute_task(ctx, launcher));
      processed += cnt;
    }

    // Barrier per level
    for (auto& f : futures) f.get_void_result();
  }

  // Final sift-down at root
  sift_down<int*>(data, data + n, std::less<int>(), n, data);
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
  std::size_t begin_last = (v.size() > print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = begin_last; i < v.size(); ++i) {
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

void top_level_task(const Task* /*task*/,
                    const std::vector<PhysicalRegion>& /*regions*/,
                    Context ctx,
                    Runtime* runtime) {
  const InputArgs& command_args = Runtime::get_input_args();

  boost::program_options::options_description cmdline("usage: heaps [options]");
  cmdline.add_options()
      ("vector_size",
       boost::program_options::value<std::size_t>()->default_value(25),
       "size of vector")
      ("chunk_size",
       boost::program_options::value<std::size_t>()->default_value(0),
       "amount of work per task")
      ("help", "print help");

  boost::program_options::variables_map vm;
  auto parsed = boost::program_options::command_line_parser(command_args.argc, command_args.argv)
                    .options(cmdline)
                    .allow_unregistered() // ignore Legion runtime flags (-lg:...)
                    .run();
  boost::program_options::store(parsed, vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << cmdline << "\n";
    return;
  }

  std::size_t vector_size = vm["vector_size"].as<std::size_t>();
  std::size_t chunk_size = vm["chunk_size"].as<std::size_t>();

  std::vector<int> v(vector_size);
  std::iota(v.begin(), v.end(), 0);

  if (chunk_size == 0) {
    std::size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 1;
    chunk_size = std::max<std::size_t>(1, vector_size / threads);
  }

  make_heap_legion(v, chunk_size, ctx, runtime);
  write_heap_characteristics(v);
}

int main(int argc, char* argv[]) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(SIFT_RANGE_TASK_ID, "sift_range_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<sift_range_task>(registrar, "sift_range_task");
  }

  return Runtime::start(argc, argv);
}
