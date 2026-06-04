#include <legion.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 1,
    SIFT_DOWN_RANGE_TASK_ID = 2
};

struct LevelTaskArgs {
    std::size_t level_start; // start index for this launched level chunking
    std::size_t items;       // number of heap nodes to process in this level range
    std::size_t chunk_size;  // number of nodes per Legion task
    std::size_t len;         // total heap length
};

// Shared heap storage for child tasks (single-address-space execution assumption).
static std::vector<int>* g_heap_vector = nullptr;

static std::size_t highest_power_of_two_leq(std::size_t x) {
    if (x == 0) return 0;
    std::size_t p = 1;
    while ((p << 1) > p && (p << 1) <= x) {
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
}

void sift_down_range_task(const Task* task,
                          const std::vector<PhysicalRegion>&,
                          Context,
                          Runtime*) {
    if (g_heap_vector == nullptr || task->args == nullptr) return;

    const auto* args = static_cast<const LevelTaskArgs*>(task->args);
    const coord_t point = task->index_point[0];
    if (point < 0) return;

    std::size_t cnt = static_cast<std::size_t>(point) * args->chunk_size;
    if (cnt >= args->items) return;

    std::size_t count = std::min(args->chunk_size, args->items - cnt);
    std::size_t start_index = args->level_start - cnt;

    auto& v = *g_heap_vector;
    using diff_t = typename std::iterator_traits<std::vector<int>::iterator>::difference_type;

    std::less<int> pred;
    sift_down_range(v.begin(),
                    v.end(),
                    pred,
                    static_cast<diff_t>(args->len),
                    v.begin() + static_cast<diff_t>(start_index),
                    count);
}

static void make_heap_legion(std::vector<int>& v,
                             std::size_t chunk_size,
                             Context ctx,
                             Runtime* runtime) {
    using diff_t = typename std::iterator_traits<std::vector<int>::iterator>::difference_type;
    const diff_t n = static_cast<diff_t>(v.size());

    if (n <= 1) return;
    if (chunk_size == 0) chunk_size = 1;

    // Bottom-up level progression: process one tree depth per barrier.
    for (diff_t start = (n - 2) / 2; start > 0; ) {
        diff_t end_level =
            static_cast<diff_t>(highest_power_of_two_leq(static_cast<std::size_t>(start) + 1)) - 2;

        std::size_t items = static_cast<std::size_t>(start - end_level);

        std::size_t level_chunk = chunk_size;
        if (level_chunk > items) {
            level_chunk = std::max<std::size_t>(1, items / 2);
        }
        if (level_chunk == 0) level_chunk = 1;

        std::size_t num_tasks = (items + level_chunk - 1) / level_chunk;

        LevelTaskArgs args{
            static_cast<std::size_t>(start),
            items,
            level_chunk,
            static_cast<std::size_t>(n)
        };

        Rect<1> launch_rect(Point<1>(0),
                            Point<1>(static_cast<coord_t>(num_tasks - 1)));
        IndexTaskLauncher launcher(
            SIFT_DOWN_RANGE_TASK_ID,
            Domain(launch_rect),
            TaskArgument(&args, sizeof(args)),
            ArgumentMap()
        );

        FutureMap fm = runtime->execute_index_space(ctx, launcher);
        fm.wait_all_results(); // barrier per level

        start = end_level;
    }

    std::less<int> pred;
    sift_down(v.begin(), v.end(), pred, n, v.begin());
}

static std::size_t parse_size_or_default(const std::string& s, std::size_t defval) {
    try {
        return static_cast<std::size_t>(std::stoull(s));
    } catch (...) {
        return defval;
    }
}

static void parse_cli_args(const InputArgs& inputs,
                           std::size_t& vector_size,
                           std::size_t& chunk_size) {
    for (int i = 1; i < inputs.argc; ++i) {
        std::string arg(inputs.argv[i]);

        if (arg == "--vector_size") {
            if (i + 1 < inputs.argc) {
                vector_size = parse_size_or_default(inputs.argv[++i], vector_size);
            }
        } else if (arg.rfind("--vector_size=", 0) == 0) {
            vector_size = parse_size_or_default(arg.substr(std::string("--vector_size=").size()),
                                                vector_size);
        } else if (arg == "--chunk_size") {
            if (i + 1 < inputs.argc) {
                chunk_size = parse_size_or_default(inputs.argv[++i], chunk_size);
            }
        } else if (arg.rfind("--chunk_size=", 0) == 0) {
            chunk_size = parse_size_or_default(arg.substr(std::string("--chunk_size=").size()),
                                               chunk_size);
        }
    }
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
    std::size_t vector_size = 25;
    std::size_t chunk_size = 0;

    InputArgs inputs = Runtime::get_input_args();
    parse_cli_args(inputs, vector_size, chunk_size);

    std::vector<int> v(vector_size);
    std::iota(v.begin(), v.end(), 0);

    if (chunk_size == 0) {
        std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
        chunk_size = std::max<std::size_t>(1, (vector_size == 0 ? 1 : (vector_size / threads)));
    }

    g_heap_vector = &v;
    make_heap_legion(v, chunk_size, ctx, runtime);
    g_heap_vector = nullptr;

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
        TaskVariantRegistrar registrar(SIFT_DOWN_RANGE_TASK_ID, "sift_down_range_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<sift_down_range_task>(registrar,
                                                                "sift_down_range_task");
    }

    return Runtime::start(argc, argv);
}
