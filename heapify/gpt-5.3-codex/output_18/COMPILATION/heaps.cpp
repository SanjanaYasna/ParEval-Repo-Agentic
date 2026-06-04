#include "legion.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs : TaskID {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_DOWN_CHUNK_TASK_ID,
};

struct SiftChunkArgs {
  int* data;
  std::size_t len;
  std::size_t start_index;  // right-most parent index in this chunk
  std::size_t count;        // number of parent nodes to process, moving left
};

struct ProgramOptions {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
  bool help = false;
};

static bool parse_size_t(const std::string& s, std::size_t& out) {
  try {
    std::size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos);
    if (pos != s.size()) return false;
    out = static_cast<std::size_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

static ProgramOptions parse_options(const InputArgs& args) {
  ProgramOptions opts;

  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);

    if (a == "--help" || a == "-h") {
      opts.help = true;
      continue;
    }

    if (a == "--vector_size") {
      if (i + 1 < args.argc) {
        std::size_t v = 0;
        if (parse_size_t(args.argv[++i], v)) opts.vector_size = v;
      }
      continue;
    }

    if (a.rfind("--vector_size=", 0) == 0) {
      std::size_t v = 0;
      if (parse_size_t(a.substr(std::string("--vector_size=").size()), v))
        opts.vector_size = v;
      continue;
    }

    if (a == "--chunk_size") {
      if (i + 1 < args.argc) {
        std::size_t c = 0;
        if (parse_size_t(args.argv[++i], c)) opts.chunk_size = c;
      }
      continue;
    }

    if (a.rfind("--chunk_size=", 0) == 0) {
      std::size_t c = 0;
      if (parse_size_t(a.substr(std::string("--chunk_size=").size()), c))
        opts.chunk_size = c;
      continue;
    }

    // Ignore unknown options (e.g. Legion runtime flags like -ll:cpu).
  }

  return opts;
}

static void print_usage(const char* prog) {
  std::cout << "usage: " << (prog ? prog : "heaps") << " [options]\n\n"
            << "options:\n"
            << "  --vector_size <int>   size of vector (default: 25)\n"
            << "  --chunk_size <int>    amount of work per task (default: 0 => auto)\n"
            << "  --help                print this help message\n";
}

static std::size_t floor_log2(std::size_t x) {
  std::size_t r = 0;
  while ((r + 1) < (sizeof(std::size_t) * 8) && ((std::size_t(1) << (r + 1)) <= x)) {
    ++r;
  }
  return r;
}

static void write_heap_characteristics(const std::vector<int>& v) {
  std::ofstream outFile("heaps.txt");

  std::size_t first_count = std::min<std::size_t>(v.size(), 10);
  outFile << "First " << first_count << " elements: ";
  for (std::size_t i = 0; i < first_count; ++i) {
    outFile << v[i];
    if (i + 1 < first_count) outFile << " ";
  }

  std::size_t last_count = std::min<std::size_t>(v.size(), 10);
  std::size_t last_start = v.size() - last_count;
  outFile << "\nLast " << last_count << " elements: ";
  for (std::size_t i = last_start; i < v.size(); ++i) {
    outFile << v[i];
    if (i + 1 < v.size()) outFile << " ";
  }

  long long sum = std::accumulate(v.begin(), v.end(), 0LL);
  outFile << "\nSum of all elements: " << sum << "\n";

  if (!v.empty()) {
    outFile << "Root (max) element: " << v[0] << "\n";
  }

  bool is_valid_heap = std::is_heap(v.begin(), v.end());
  outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";
}

static void sift_down_chunk_task(const Task* task,
                                 const std::vector<PhysicalRegion>& /*regions*/,
                                 Context /*ctx*/,
                                 Runtime* /*runtime*/) {
  const auto* args = static_cast<const SiftChunkArgs*>(task->args);
  if (args == nullptr || args->data == nullptr || args->len <= 1 || args->count == 0) return;

  int* first = args->data;
  int* last = first + args->len;
  std::less<int> pred;

  using diff_t = typename std::iterator_traits<int*>::difference_type;
  diff_t len = static_cast<diff_t>(args->len);

  for (std::size_t i = 0; i < args->count; ++i) {
    std::size_t idx = args->start_index - i;
    heap_legion::sift_down<int*, std::less<int>>(first, last, pred, len, first + idx);
  }
}

static void legion_make_heap(std::vector<int>& v,
                             std::size_t chunk_size,
                             Context ctx,
                             Runtime* runtime) {
  const std::size_t n = v.size();
  if (n <= 1) return;

  const std::size_t last_internal = (n - 2) / 2;

  // Process internal levels bottom-up, excluding root level (depth 0).
  if (last_internal >= 1) {
    const std::size_t max_depth = floor_log2(last_internal + 1);

    for (std::ptrdiff_t depth = static_cast<std::ptrdiff_t>(max_depth); depth >= 1; --depth) {
      const std::size_t d = static_cast<std::size_t>(depth);
      const std::size_t level_left = (std::size_t(1) << d) - 1;
      const std::size_t level_right = std::min(last_internal, (std::size_t(1) << (d + 1)) - 2);

      if (level_right < level_left) continue;

      const std::size_t items = level_right - level_left + 1;
      const std::size_t local_chunk = std::max<std::size_t>(1, std::min(chunk_size, items));

      std::vector<Future> futures;
      futures.reserve((items + local_chunk - 1) / local_chunk);

      std::size_t processed = 0;
      while (processed < items) {
        const std::size_t this_chunk = std::min(local_chunk, items - processed);

        SiftChunkArgs args;
        args.data = v.data();
        args.len = n;
        args.start_index = level_right - processed;
        args.count = this_chunk;

        TaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID, TaskArgument(&args, sizeof(args)));
        futures.emplace_back(runtime->execute_task(ctx, launcher));

        processed += this_chunk;
      }

      // Barrier between levels.
      for (Future& f : futures) {
        f.get_void_result();
      }
    }
  }

  // Root node.
  std::less<int> pred;
  using diff_t = typename std::iterator_traits<int*>::difference_type;
  heap_legion::sift_down<int*, std::less<int>>(v.data(), v.data() + n, pred, static_cast<diff_t>(n), v.data());
}

static void top_level_task(const Task*,
                           const std::vector<PhysicalRegion>&,
                           Context ctx,
                           Runtime* runtime) {
  InputArgs in = Runtime::get_input_args();
  ProgramOptions opts = parse_options(in);

  if (opts.help) {
    print_usage((in.argc > 0) ? in.argv[0] : "heaps");
    return;
  }

  std::vector<int> v(opts.vector_size);
  std::iota(v.begin(), v.end(), 0);

  std::size_t chunk_size = opts.chunk_size;
  if (chunk_size == 0) {
    std::size_t threads = std::max<unsigned>(1, std::thread::hardware_concurrency());
    chunk_size = std::max<std::size_t>(1, opts.vector_size / threads);
  }

  legion_make_heap(v, chunk_size, ctx, runtime);
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
    TaskVariantRegistrar registrar(SIFT_DOWN_CHUNK_TASK_ID, "sift_down_chunk_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<sift_down_chunk_task>(registrar, "sift_down_chunk_task");
  }

  return Runtime::start(argc, argv);
}
