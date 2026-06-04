// heaps.cpp — Legion translation
#include "legion.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cassert>
#include <string>

#include "sift.hpp"

using namespace Legion;

// ----------------------------------------------------------------
// Task and field IDs
// ----------------------------------------------------------------
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INIT_TASK_ID,
    SIFT_DOWN_CHUNK_TASK_ID,
};

enum FieldIDs {
    FID_VAL,
};

// ----------------------------------------------------------------
// Per-chunk arguments passed to every sift-down task
// ----------------------------------------------------------------
struct ChunkArgs {
    int64_t n;           // total number of elements
    int64_t start_idx;   // heap-array index of the right-most node in this chunk
    int64_t count;       // how many consecutive nodes (going left) to sift
};

// ----------------------------------------------------------------
// Integer floor(log2(x)) for x >= 1
// ----------------------------------------------------------------
static int64_t ilog2(int64_t x)
{
    assert(x > 0);
    int64_t r = 0;
    x >>= 1;
    while (x > 0) { x >>= 1; r++; }
    return r;
}

// ----------------------------------------------------------------
// INIT task – fills the region with 0, 1, 2, …, n-1
// ----------------------------------------------------------------
void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
    const FieldAccessor<WRITE_DISCARD, int, 1, coord_t,
          Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> bounds = dom;

    for (coord_t i = bounds.lo[0]; i <= bounds.hi[0]; i++) {
        acc[i] = static_cast<int>(i);
    }
}

// ----------------------------------------------------------------
// SIFT-DOWN-CHUNK task – sifts a contiguous range of heap nodes
// Works for both single-task launches (args in task->args) and
// index-task launches (per-point args in task->local_args).
// ----------------------------------------------------------------
void sift_down_chunk_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    ChunkArgs args;
    if (task->local_arglen >= sizeof(ChunkArgs)) {
        memcpy(&args, task->local_args, sizeof(ChunkArgs));
    } else {
        assert(task->arglen >= sizeof(ChunkArgs));
        memcpy(&args, task->args, sizeof(ChunkArgs));
    }

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
          Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    // Obtain a raw int* – the 1-D affine accessor guarantees
    // contiguous, unit-stride storage, so pointer arithmetic is valid.
    int *first = acc.ptr(Point<1>(0));

    sift_down_range<int>(first, std::less<int>(),
                         static_cast<long long>(args.n),
                         static_cast<long long>(args.start_idx),
                         static_cast<std::size_t>(args.count));
}

// ----------------------------------------------------------------
// Write heap diagnostics – same format as the HPX version
// ----------------------------------------------------------------
static void write_heap_characteristics(const std::vector<int>& v)
{
    std::ofstream outFile("heaps.txt");

    std::size_t print_count = std::min(v.size(), std::size_t(10));
    outFile << "First " << print_count << " elements: ";
    for (std::size_t i = 0; i < print_count; ++i) {
        outFile << v[i];
        if (i < print_count - 1) outFile << " ";
    }
    outFile << "\n";

    std::size_t last_count = std::min(v.size(), std::size_t(10));
    outFile << "Last " << last_count << " elements: ";
    for (std::size_t i = v.size() - last_count; i < v.size(); ++i) {
        outFile << v[i];
        if (i < v.size() - 1) outFile << " ";
    }
    outFile << "\n";

    long long sum = std::accumulate(v.begin(), v.end(), 0LL);
    outFile << "Sum of all elements: " << sum << "\n";

    if (!v.empty()) {
        outFile << "Root (max) element: " << v[0] << "\n";
    }

    bool is_valid_heap = std::is_heap(v.begin(), v.end());
    outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";

    outFile.close();
}

// ----------------------------------------------------------------
// TOP-LEVEL task – orchestrates the whole computation
// ----------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---------- command-line parsing ----------------------------------
    int64_t vector_size = 25;
    int64_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        std::string arg(command_args.argv[i]);
        if (arg == "--vector_size" && i + 1 < command_args.argc) {
            vector_size = atol(command_args.argv[++i]);
        } else if (arg.find("--vector_size=") == 0) {
            vector_size = atol(arg.substr(std::string("--vector_size=").size()).c_str());
        } else if (arg == "--chunk_size" && i + 1 < command_args.argc) {
            chunk_size = atol(command_args.argv[++i]);
        } else if (arg.find("--chunk_size=") == 0) {
            chunk_size = atol(arg.substr(std::string("--chunk_size=").size()).c_str());
        }
    }

    // automatic chunk-size: vector_size / #CPU processors
    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        int num_procs = static_cast<int>(pq.count());
        if (num_procs < 1) num_procs = 1;
        chunk_size = vector_size / num_procs;
        if (chunk_size < 1) chunk_size = 1;
    }

    int64_t n = vector_size;

    if (n <= 0) {
        std::vector<int> v;
        write_heap_characteristics(v);
        return;
    }

    // ---------- create logical region --------------------------------
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, n - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---------- initialise with iota values --------------------------
    {
        TaskLauncher launcher(INIT_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr)
            .add_field(FID_VAL));
        runtime->execute_task(ctx, launcher);
    }

    // ---------- bottom-up, level-parallel heap construction ----------
    if (n > 1) {
        int64_t start = (n - 2) / 2;
        while (start > 0) {
            // Compute level of node 'start' in the 0-indexed binary heap.
            // Node i is at level floor(log2(i+1)).
            int64_t level = ilog2(start + 1);
            int64_t level_first = (1LL << level) - 1; // first index at this level
            int64_t items = start - level_first + 1;

            int64_t level_chunk = chunk_size;
            if (level_chunk > items)
                level_chunk = items / 2;
            if (level_chunk < 1) level_chunk = 1;

            int64_t num_tasks = (items + level_chunk - 1) / level_chunk;

            if (num_tasks == 1) {
                // single task – EXCLUSIVE is sufficient
                ChunkArgs ca;
                ca.n         = n;
                ca.start_idx = start;
                ca.count     = items;
                TaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID,
                                      TaskArgument(&ca, sizeof(ca)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr)
                    .add_field(FID_VAL));
                runtime->execute_task(ctx, launcher);
            } else {
                // parallel index launch – SIMULTANEOUS coherence lets
                // all point tasks share the same physical instance
                Rect<1> launch_bounds(0, num_tasks - 1);
                IndexSpace launch_is =
                    runtime->create_index_space(ctx, launch_bounds);

                ArgumentMap arg_map;
                int64_t cnt = 0;
                for (int64_t t = 0; t < num_tasks; t++) {
                    int64_t count = (t < num_tasks - 1)
                                        ? level_chunk
                                        : items - cnt;
                    ChunkArgs ca;
                    ca.n         = n;
                    ca.start_idx = start - cnt;
                    ca.count     = count;
                    arg_map.set_point(DomainPoint(Point<1>(t)),
                                      TaskArgument(&ca, sizeof(ca)));
                    cnt += count;
                }

                IndexTaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID,
                                           launch_is,
                                           TaskArgument(NULL, 0),
                                           arg_map);
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr)
                    .add_field(FID_VAL));

                // Legion's dependence analysis on lr provides the
                // inter-level barrier automatically.
                runtime->execute_index_space(ctx, launcher);

                runtime->destroy_index_space(ctx, launch_is);
            }

            start = level_first - 1; // move to previous level
        }

        // sift down the root node
        {
            ChunkArgs ca;
            ca.n         = n;
            ca.start_idx = 0;
            ca.count     = 1;
            TaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID,
                                  TaskArgument(&ca, sizeof(ca)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr)
                .add_field(FID_VAL));
            runtime->execute_task(ctx, launcher);
        }
    }

    // ---------- read back & write diagnostics ------------------------
    {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
              Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);

        std::vector<int> v(n);
        for (int64_t i = 0; i < n; i++) {
            v[i] = acc[i];
        }
        runtime->unmap_region(ctx, pr);

        write_heap_characteristics(v);
    }

    // ---------- clean up ---------------------------------------------
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ----------------------------------------------------------------
// main – register tasks and start the Legion runtime
// ----------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar,
                                                          "top_level");
    }
    {
        TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<init_task>(registrar, "init");
    }
    {
        TaskVariantRegistrar registrar(SIFT_DOWN_CHUNK_TASK_ID,
                                       "sift_down_chunk");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<sift_down_chunk_task>(registrar,
                                                                 "sift_down_chunk");
    }

    return Runtime::start(argc, argv);
}
