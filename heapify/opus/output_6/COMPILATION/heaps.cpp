#include "legion.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cassert>

#include "sift.hpp"

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    SIFT_DOWN_RANGE_TASK_ID,
};

enum {
    FID_VAL = 101,
};

struct SiftDownArgs {
    long long n;       // total number of elements
    long long start;   // starting position (rightmost element of chunk)
    long long count;   // number of elements to sift in this chunk
};

void sift_down_range_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(SiftDownArgs));
    SiftDownArgs args;
    memcpy(&args, task->args, sizeof(SiftDownArgs));

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);
    int *base = acc.ptr(Point<1>(0));

    sift_down_range<int>(base, std::less<int>(),
                         args.n, args.start,
                         static_cast<std::size_t>(args.count));
}

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

    std::size_t last_start = v.size() >= print_count ? v.size() - print_count : 0;
    outFile << "Last " << print_count << " elements: ";
    for (std::size_t i = last_start; i < v.size(); ++i) {
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

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- Parse command-line arguments ----
    std::size_t vector_size = 25;
    std::size_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--vector_size") &&
            i + 1 < command_args.argc) {
            vector_size = std::atol(command_args.argv[++i]);
        } else if (!strcmp(command_args.argv[i], "--chunk_size") &&
                   i + 1 < command_args.argc) {
            chunk_size = std::atol(command_args.argv[++i]);
        }
    }

    // Default chunk_size: vector_size / number-of-CPU-processors
    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        std::size_t threads = pq.count();
        if (threads == 0) threads = 1;
        chunk_size = vector_size / threads;
        if (chunk_size == 0) chunk_size = 1;
    }

    if (vector_size == 0) {
        write_heap_characteristics(std::vector<int>());
        return;
    }

    // ---- Create logical region ----
    IndexSpaceT<1> is = runtime->create_index_space(ctx,
        Rect<1>(0, static_cast<coord_t>(vector_size - 1)));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---- Initialise with iota: 0, 1, 2, …, vector_size-1 ----
    {
        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
        for (std::size_t i = 0; i < vector_size; i++)
            acc[i] = static_cast<int>(i);
        runtime->unmap_region(ctx, pr);
    }

    // ---- Parallel make_heap (level-parallel, bottom-up) ----
    long long n = static_cast<long long>(vector_size);

    if (n > 1) {
        long long last_parent = (n - 2) / 2;

        // Integer floor(log2(x)) for x >= 1
        auto ilog2 = [](long long x) -> int {
            int r = 0;
            while (x > 1) { x >>= 1; r++; }
            return r;
        };

        // Process levels from bottom to top (excluding root level 0).
        // Level lev contains nodes [2^lev - 1, 2^(lev+1) - 2].
        // At the deepest non-leaf level, we only go up to last_parent.
        int max_level = ilog2(last_parent + 1);

        for (int lev = max_level; lev >= 1; lev--) {
            long long level_first = (1LL << lev) - 1;
            long long level_last  = (1LL << (lev + 1)) - 2;
            long long right = std::min(last_parent, level_last);
            long long left  = level_first;

            long long items = right - left + 1;
            if (items <= 0) continue;

            // Adapt chunk_size for this level (use local copy)
            std::size_t cs = chunk_size;
            if (cs > static_cast<std::size_t>(items))
                cs = std::max(static_cast<std::size_t>(1),
                              static_cast<std::size_t>(items) / 2);

            std::vector<Future> futures;
            futures.reserve(static_cast<std::size_t>(items) / cs + 1);

            // Schedule work in cs-sized pieces, walking right to left
            long long pos = right;
            while (pos >= left) {
                long long cnt = std::min(static_cast<long long>(cs),
                                         pos - left + 1);

                SiftDownArgs args;
                args.n     = n;
                args.start = pos;
                args.count = cnt;

                TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                    TaskArgument(&args, sizeof(SiftDownArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                futures.push_back(runtime->execute_task(ctx, launcher));

                pos -= cnt;
            }

            // Synchronise: all tasks at this level must finish
            // before we proceed to the next (upper) level.
            for (auto &f : futures)
                f.get_void_result();
        }

        // Sift down the root node (level 0)
        {
            SiftDownArgs args;
            args.n     = n;
            args.start = 0;
            args.count = 1;

            TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                TaskArgument(&args, sizeof(SiftDownArgs)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher).get_void_result();
        }
    }

    // ---- Read back and write heap characteristics ----
    {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);

        std::vector<int> v(vector_size);
        for (std::size_t i = 0; i < vector_size; i++)
            v[i] = acc[i];
        runtime->unmap_region(ctx, pr);

        write_heap_characteristics(v);
    }

    // ---- Cleanup ----
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
        Runtime::preregister_task_variant<top_level_task>(
            registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(SIFT_DOWN_RANGE_TASK_ID,
                                       "sift_down_range");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<sift_down_range_task>(
            registrar, "sift_down_range");
    }

    return Runtime::start(argc, argv);
}
