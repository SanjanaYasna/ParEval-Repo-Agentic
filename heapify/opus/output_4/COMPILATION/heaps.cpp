// heaps.cpp — Legion translation of the HPX level-parallel make_heap
#include "legion.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cassert>
#include <thread>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    SIFT_DOWN_RANGE_TASK_ID,
};

enum FieldIDs {
    FID_VAL,
};

struct SiftDownRangeArgs {
    int64_t  n;          // total number of elements
    int64_t  start_idx;  // heap-array index of the first node to sift
    uint64_t count;      // how many consecutive nodes to sift
};

// ------------------------------------------------------------------ //
// Task: sift_down_range – called once per chunk inside an IndexLaunch
// ------------------------------------------------------------------ //
void sift_down_range_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(task->local_arglen == sizeof(SiftDownRangeArgs));
    SiftDownRangeArgs args;
    memcpy(&args, task->local_args, sizeof(SiftDownRangeArgs));

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    int *first = acc.ptr(Point<1>(0));

    // Use the index-based sift_down_range from sift.hpp
    sift_down_range(first, static_cast<long long>(args.n), std::less<int>(),
                    static_cast<long long>(args.start_idx),
                    static_cast<std::size_t>(args.count));
}

// ------------------------------------------------------------------ //
// Helper: write heap diagnostics (identical to the original)
// ------------------------------------------------------------------ //
void write_heap_characteristics(const std::vector<int> &v)
{
    std::ofstream outFile("heaps.txt");

    std::size_t print_count = std::min(v.size(), std::size_t(10));
    outFile << "First " << print_count << " elements: ";
    for (std::size_t i = 0; i < print_count; ++i) {
        outFile << v[i];
        if (i < print_count - 1) outFile << " ";
    }
    outFile << "\n";

    outFile << "Last " << print_count << " elements: ";
    for (std::size_t i = v.size() - 10; i < v.size(); ++i) {
        outFile << v[i];
        if (i < v.size() - 1) outFile << " ";
    }
    outFile << "\n";

    long long sum = std::accumulate(v.begin(), v.end(), 0LL);
    outFile << "Sum of all elements: " << sum << "\n";

    if (!v.empty())
        outFile << "Root (max) element: " << v[0] << "\n";

    bool is_valid_heap = std::is_heap(v.begin(), v.end());
    outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";

    outFile.close();
}

// ------------------------------------------------------------------ //
// Top-level task
// ------------------------------------------------------------------ //
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing (mirrors the HPX version) ----
    std::size_t vector_size = 25;
    std::size_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--vector_size") == 0 &&
            i + 1 < command_args.argc) {
            vector_size = static_cast<std::size_t>(std::atol(command_args.argv[++i]));
        } else if (strcmp(command_args.argv[i], "--chunk_size") == 0 &&
                   i + 1 < command_args.argc) {
            chunk_size = static_cast<std::size_t>(std::atol(command_args.argv[++i]));
        }
    }

    if (chunk_size == 0) {
        std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
        chunk_size = vector_size / threads;
        if (chunk_size == 0) chunk_size = 1;
    }

    // ---- create logical region ----
    IndexSpace is = runtime->create_index_space(ctx,
        Rect<1>(0, static_cast<coord_t>(vector_size) - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
        alloc.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---- initialise with iota values (inline mapping) ----
    {
        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        int *base = acc.ptr(Point<1>(0));
        for (std::size_t i = 0; i < vector_size; i++)
            base[i] = static_cast<int>(i);

        runtime->unmap_region(ctx, pr);
    }

    // ---- build the heap: iterate over levels bottom-up ----
    const int64_t n = static_cast<int64_t>(vector_size);

    if (n > 1) {
        for (int64_t start = (n - 2) / 2; start > 0;
             start = static_cast<int64_t>(
                         pow(2.0,
                             static_cast<double>(
                                 static_cast<int64_t>(
                                     log2(static_cast<double>(start)))))) - 2)
        {
            int64_t end_level = static_cast<int64_t>(
                pow(2.0,
                    static_cast<double>(
                        static_cast<int64_t>(
                            log2(static_cast<double>(start)))))) - 2;

            std::size_t items = static_cast<std::size_t>(start - end_level);

            std::size_t cs = chunk_size;
            if (cs > items)
                cs = items / 2;
            if (cs == 0)
                cs = 1;

            // Build per-chunk argument list
            std::vector<SiftDownRangeArgs> chunk_args;
            std::size_t cnt = 0;

            while (cnt + cs < items) {
                SiftDownRangeArgs a;
                a.n         = n;
                a.start_idx = start - static_cast<int64_t>(cnt);
                a.count     = static_cast<uint64_t>(cs);
                chunk_args.push_back(a);
                cnt += cs;
            }
            if (cnt < items) {
                SiftDownRangeArgs a;
                a.n         = n;
                a.start_idx = start - static_cast<int64_t>(cnt);
                a.count     = static_cast<uint64_t>(items - cnt);
                chunk_args.push_back(a);
            }

            const int num_chunks = static_cast<int>(chunk_args.size());

            // Populate per-point argument map
            ArgumentMap arg_map;
            for (int i = 0; i < num_chunks; i++) {
                arg_map.set_point(DomainPoint(i),
                    TaskArgument(&chunk_args[i], sizeof(SiftDownRangeArgs)));
            }

            // Launch one index-space task per chunk; SIMULTANEOUS coherence
            // lets all point tasks access the same physical instance in
            // parallel (safe because subtrees at the same level are disjoint).
            IndexLauncher idx_launcher(SIFT_DOWN_RANGE_TASK_ID,
                Rect<1>(0, num_chunks - 1),
                TaskArgument(NULL, 0), arg_map);

            idx_launcher.add_region_requirement(
                RegionRequirement(lr, 0 /*projection ID*/,
                                  READ_WRITE, SIMULTANEOUS, lr));
            idx_launcher.region_requirements[0].add_field(FID_VAL);

            // Blocking call — acts as the per-level barrier
            runtime->execute_index_space(ctx, idx_launcher);
        }

        // ---- sift down the root node (inline mapping) ----
        {
            RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1, coord_t,
                Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
            int *base = acc.ptr(Point<1>(0));
            sift_down(base, static_cast<long long>(n), std::less<int>(), 0LL);

            runtime->unmap_region(ctx, pr);
        }
    }

    // ---- read back and write diagnostics ----
    {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);

        std::vector<int> v(vector_size);
        for (std::size_t i = 0; i < vector_size; i++)
            v[i] = acc[i];

        runtime->unmap_region(ctx, pr);

        write_heap_characteristics(v);
    }

    // ---- clean up ----
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ------------------------------------------------------------------ //
// main – register tasks and start the Legion runtime
// ------------------------------------------------------------------ //
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(SIFT_DOWN_RANGE_TASK_ID, "sift_down_range");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<sift_down_range_task>(registrar,
                                                                 "sift_down_range");
    }

    return Runtime::start(argc, argv);
}
