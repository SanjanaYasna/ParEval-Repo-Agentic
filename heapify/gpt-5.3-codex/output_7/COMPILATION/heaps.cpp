#include "legion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs : TaskID {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID = 2
};

struct HeapConfig {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
};

struct SiftTaskArgs {
  int* data;
  std::ptrdiff_t len;
  std::ptrdiff_t start_index;
  std::size_t count;
};

static inline bool starts_with(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

static bool parse_size_t(const std::string& text, std::size_t& out) {
  try {
    std::size_t pos = 0;
    unsigned long long v = std::stoull(text, &pos);
    if (pos != text.size()) return false;
    out = static_cast<std::size_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

static HeapConfig parse_heap_config(const InputArgs& input_args) {
  HeapConfig cfg;

  for (int i = 1; i < input_args.argc; ++i) {
    const std::string arg(input_args.argv[i]);

    if (arg == "--vector_size") {
      if (i + 1 < input_args.argc) {
        std::size_t val = cfg.vector_size;
        if (parse_size_t(input_args.argv[++i], val)) cfg.vector_size = val;
      }
    } else if (starts_with(arg, "--vector_size=")) {
      std::size_t val = cfg.vector_size;
      if (parse_size_t(arg.substr(std::string("--vector_size=").size()), val)) {
        cfg.vector_size = val;
      }
    } else if (arg == "--chunk_size") {
      if (i + 1 < input_args.argc) {
        std::size_t val = cfg.chunk_size;
        if (parse_size_t(input_args.argv[++i], val)) cfg.chunk_size = val;
      }
    } else if (starts_with(arg, "--chunk_size=")) {
      std::size_t val = cfg.chunk_size;
      if (parse_size_t(arg.substr(std::string("--chunk_size=").size()), val)) {
        cfg.chunk_size = val;
      }
    }
    // Ignore all other options (including Legion runtime flags).
  }

  return cfg;
}

static std::ptrdiff_t floor_power_of_two(std::ptrdiff_t x) {
  if (x <= 0) return 0;
  std::ptrdiff_t p = 1;
  while (p <= (x / 2)) {
    p <<= 1;
  }
  return p;
}

static void write_heap_characteristics(const std::vector<int>& v) {
  std::ofstream outFile("heaps.txt");

  std::size_t print_count = std::min<std::size_t>(v.size(), 10);
  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i) {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\n";
  outFile << "Last " << print_count << " elements: ";
  std::size_t start = (v.size() > print_count) ? (v.size() - print_count) : 0;
  for (std::size_t i = start; i < v.size(); ++i) {
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

void sift_range_task(const Task* task,
                     const std::vector<PhysicalRegion>&,
                     Context,
                     Runtime*) {
  const SiftTaskArgs* args = static_cast<const SiftTaskArgs*>(task->args);
  if (args == nullptr || args->data == nullptr || args->count == 0 || args->len <= 1) return;

  heap_legion::sift_down_range(args->data, args->len, args->start_index, args->count);
}

static void make_heap_legion(std::vector<int>& v,
                             std::size_t chunk_size,
                             Context ctx,
                             Runtime* runtime) {
  using difference_type = std::ptrdiff_t;
  const difference_type n = static_cast<difference_type>(v.size());
  if (n <= 1) return;

  int* data = v.data();
  difference_type level_hi = (n - 2) / 2;  // last internal node

  while (true) {
    const difference_type level_lo = floor_power_of_two(level_hi + 1) - 1;
    const std::size_t items = static_cast<std::size_t>(level_hi - level_lo + 1);

    std::size_t local_chunk = (chunk_size == 0) ? items : chunk_size;
    if (local_chunk == 0) local_chunk = 1;
    if (local_chunk > items) local_chunk = items;

    std::vector<Future> workitems;
    workitems.reserve((items + local_chunk - 1) / local_chunk);

    for (std::size_t offset = 0; offset < items; offset += local_chunk) {
      const std::size_t count = std::min(local_chunk, items - offset);

      SiftTaskArgs args{
          data,
          n,
          level_hi - static_cast<difference_type>(offset),
          count
      };

      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      workitems.emplace_back(runtime->execute_task(ctx, launcher));
    }

    for (Future& f : workitems) {
      f.get_void_result(); // level barrier
    }

    if (level_lo == 0) break;
    level_hi = level_lo - 1;
  }
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
  const InputArgs& args = Runtime::get_input_args();
  HeapConfig cfg = parse_heap_config(args);

  std::vector<int> v(cfg.vector_size);
  std::iota(v.begin(), v.end(), 0);

  std::size_t chunk_size = cfg.chunk_size;
  if (chunk_size == 0) {
    std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
    chunk_size = cfg.vector_size / threads;
    if (chunk_size == 0) chunk_size = 1;
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
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_range_task>(registrar, "sift_range_task");
  }

  return Runtime::start(argc, argv);
}
