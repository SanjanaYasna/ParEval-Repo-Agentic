#include "legion.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SIFT_RANGE_TASK_ID
};

// Shared vector used by launched Legion tasks (single-address-space assumption)
static std::atomic<std::vector<int>*> g_heap_data{nullptr};

struct SiftRangeArgs {
  long long len;
  long long start_index;
  std::size_t count;
};

static inline std::size_t highest_power_of_two_leq(std::size_t x) {
  std::size_t p = 1;
  while ((p << 1) != 0 && (p << 1) <= x) {
    p <<= 1;
  }
  return p;
}

static inline std::size_t level_leftmost_index(std::size_t idx) {
  // Nodes on level l are [2^l - 1, 2^(l+1) - 2]
  return highest_power_of_two_leq(idx + 1) - 1;
}

static bool parse_size_t_str(const char* s, std::size_t& out) {
  if (s == nullptr || *s == '\0') return false;

  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return false;
  if (value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
    return false;

  out = static_cast<std::size_t>(value);
  return true;
}

void sift_range_task(const Task* task,
                     const std::vector<PhysicalRegion>& /*regions*/,
                     Context /*ctx*/,
                     Runtime* /*runtime*/) {
  auto* heap_ptr = g_heap_data.load(std::memory_order_acquire);
  if (heap_ptr == nullptr) return;

  const auto* args = static_cast<const SiftRangeArgs*>(task->args);
  if (args == nullptr) return;

  auto& v = *heap_ptr;
  using diff_t = std::vector<int>::difference_type;

  diff_t len = static_cast<diff_t>(args->len);
  diff_t start = static_cast<diff_t>(args->start_index);

  if (v.empty() || len <= 1 || start < 0 || static_cast<std::size_t>(start) >= v.size())
    return;

  sift_down_range(v.begin(), v.end(), std::less<int>{}, len, v.begin() + start,
                  args->count);
}

void make_heap_legion(std::vector<int>& v, std::size_t chunk_size, Context ctx,
                      Runtime* runtime) {
  using diff_t = std::vector<int>::difference_type;
  const diff_t n = static_cast<diff_t>(v.size());

  if (n <= 1) return;

  diff_t last_parent = (n - 2) / 2;

  // Process complete/partial levels bottom-up, excluding root.
  while (last_parent > 0) {
    diff_t level_left =
        static_cast<diff_t>(level_leftmost_index(static_cast<std::size_t>(last_parent)));
    if (level_left < 1) level_left = 1;

    const std::size_t items = static_cast<std::size_t>(last_parent - level_left + 1);
    if (items == 0) break;

    std::size_t level_chunk = (chunk_size == 0) ? 1 : chunk_size;
    if (level_chunk > items) level_chunk = items;

    std::vector<Future> futures;
    futures.reserve((items + level_chunk - 1) / level_chunk);

    std::size_t cnt = 0;
    while (cnt < items) {
      const std::size_t this_count = std::min(level_chunk, items - cnt);
      SiftRangeArgs args{static_cast<long long>(n),
                         static_cast<long long>(last_parent - static_cast<diff_t>(cnt)),
                         this_count};
      TaskLauncher launcher(SIFT_RANGE_TASK_ID, TaskArgument(&args, sizeof(args)));
      futures.emplace_back(runtime->execute_task(ctx, launcher));
      cnt += this_count;
    }

    for (auto& f : futures) {
      f.get_void_result();  // per-level synchronization barrier
    }

    last_parent = level_left - 1;
  }

  // Final sift for root
  sift_down(v.begin(), v.end(), std::less<int>{}, n, v.begin());
}

void write_heap_characteristics(const std::vector<int>& v) {
  std::ofstream outFile("heaps.txt");
  if (!outFile) return;

  std::size_t print_count = std::min<std::size_t>(v.size(), 10);

  outFile << "First " << print_count << " elements: ";
  for (std::size_t i = 0; i < print_count; ++i) {
    outFile << v[i];
    if (i + 1 < print_count) outFile << " ";
  }

  outFile << "\nLast " << print_count << " elements: ";
  std::size_t last_start = (v.size() > print_count) ? (v.size() - print_count) : 0;
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

void parse_options(std::size_t& vector_size, std::size_t& chunk_size) {
  vector_size = 25;
  chunk_size = 0;

  const InputArgs& args = Runtime::get_input_args();

  for (int i = 1; i < args.argc; ++i) {
    std::string token(args.argv[i]);

    if (token == "--vector_size") {
      if (i + 1 < args.argc) {
        std::size_t tmp = 0;
        if (parse_size_t_str(args.argv[i + 1], tmp)) vector_size = tmp;
        ++i;
      }
      continue;
    }

    if (token == "--chunk_size") {
      if (i + 1 < args.argc) {
        std::size_t tmp = 0;
        if (parse_size_t_str(args.argv[i + 1], tmp)) chunk_size = tmp;
        ++i;
      }
      continue;
    }

    const std::string vprefix = "--vector_size=";
    if (token.rfind(vprefix, 0) == 0) {
      std::size_t tmp = 0;
      if (parse_size_t_str(token.c_str() + vprefix.size(), tmp)) vector_size = tmp;
      continue;
    }

    const std::string cprefix = "--chunk_size=";
    if (token.rfind(cprefix, 0) == 0) {
      std::size_t tmp = 0;
      if (parse_size_t_str(token.c_str() + cprefix.size(), tmp)) chunk_size = tmp;
      continue;
    }
  }
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
  std::size_t vector_size = 25;
  std::size_t chunk_size = 0;
  parse_options(vector_size, chunk_size);

  std::vector<int> v(vector_size);
  std::iota(v.begin(), v.end(), 0);

  if (chunk_size == 0) {
    std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    chunk_size = vector_size / threads;
    if (chunk_size == 0) chunk_size = 1;
  }

  g_heap_data.store(&v, std::memory_order_release);
  make_heap_legion(v, chunk_size, ctx, runtime);
  g_heap_data.store(nullptr, std::memory_order_release);

  write_heap_characteristics(v);
}

int main(int argc, char** argv) {
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
