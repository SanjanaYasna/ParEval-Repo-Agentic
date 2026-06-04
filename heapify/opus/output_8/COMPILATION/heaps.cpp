//source: https://github.com/Syntaf/heapify/tree/master
// Translated from HPX to the Legion execution model.
#include "legion.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cassert>

#include "sift.hpp"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SIFT_DOWN_RANGE_TASK_ID = 1,
};

enum FieldIDs {
    FID_VAL = 0,
};

// Arguments passed to each sift-down chunk task
struct SiftDownRangeArgs {
    int64_t n;            // total number of elements
    int64_t start_offset; // index of the first node this chunk sifts
    int64_t count;        // how many consecutive nodes to sift
};

// -----------------------------------------------------------------------
// Task: sift_down_range_task
//   Obtains a raw pointer into the shared physical instance and calls the
//   existing sift_down_range helper from sift.hpp.
// -----------------------------------------------------------------------
void sift_down_range_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(SiftDownRangeArgs));
    SiftDownRangeArgs args;
    memcpy(&args, task->args, sizeof(SiftDownRangeArgs));

    // Obtain the dense pointer to the region's data
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);
    int *data = acc.ptr(rect.lo);

    int *first = data;
    int *last  = data + args.n;
    int *start = data + args.start_offset;

    // Reuse the original sift_down_range from sift.hpp
    sift_down_range(first, last, std::less<int>(), args.n, start,
                    static_cast<std::size_t>(args.count));
}

// -----------------------------------------------------------------------
// write_heap_characteristics – identical logic to the HPX version
// -----------------------------------------------------------------------
void write_heap_characteristics(const int *v, std::size_t size)
{
    std::ofstream outFile("heaps.txt");

    std::size_t print_count = std::min(size, std::size_t(10));
    outFile << "First " << print_count << " elements: ";
    for (std::size_t i = 0; i < print_count; ++i) {
        outFile << v[i];
        if (i < print_count - 1) outFile << " ";
    }
    outFile << "\n";

    std::size_t last_count = std::min(size, std::size_t(10));
    std::size_t last_start = size >= 10 ? size - 10 : 0;
    outFile << "Last " << last_count << " elements: ";
    for (std::size_t i = last_start; i < size; ++i) {
        outFile << v[i];
        if (i < size - 1) outFile << " ";
    }
    outFile << "\n";

    long long sum = 0;
    for (std::size_t i = 0; i < size; ++i)
        sum += v[i];
    outFile << "Sum of all elements: " << sum << "\n";

    if (size > 0)
        outFile << "Root (max) element: " << v[0] << "\n";

    bool is_valid_heap = std::is_heap(v, v + size);
    outFile << "Is valid heap: " << (is_valid_heap ? "true" : "false") << "\n";

    outFile.close();
}

// -----------------------------------------------------------------------
// Top-level task – replaces hpx_main
// -----------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing (mirrors the HPX options) ----------------
    std::size_t vector_size = 25;
    std::size_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--vector_size") &&
            i + 1 < command_args.argc) {
            vector_size = static_cast<std::size_t>(std::atol(command_args.argv[++i]));
        } else if (!strcmp(command_args.argv[i], "--chunk_size") &&
                   i + 1 < command_args.argc) {
            chunk_size = static_cast<std::size_t>(std::atol(command_args.argv[++i]));
        }
    }

    // Auto-select chunk_size based on available CPU processors
    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        std::size_t threads = pq.count();
        if (threads == 0) threads = 1;
        chunk_size = vector_size / threads;
        if (chunk_size == 0) chunk_size = 1;
    }

    // Handle trivial case
    if (vector_size == 0) {
        write_heap_characteristics(nullptr, 0);
        return;
    }

    // ---- create logical region for the vector -------------------------
    IndexSpace is = runtime->create_index_space(ctx,
        Rect<1>(0, static_cast<coord_t>(vector_size) - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---- initialise with iota (0, 1, 2, …) via inline mapping --------
    {
        InlineLauncher il(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_WRITE, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        int *data = acc.ptr(0);
        for (std::size_t i = 0; i < vector_size; i++)
            data[i] = static_cast<int>(i);

        runtime->unmap_region(ctx, pr);
    }

    // ---- _make_heap ---------------------------------------------------
    // Level-parallel bottom-up heap construction, translated from the HPX
    // version.  Each level's chunks are launched as individual Legion tasks
    // with SIMULTANEOUS coherence so they may execute concurrently on the
    // shared physical instance (the tasks touch disjoint subtrees).  An
    // explicit barrier (get_void_result on every future) separates levels.
    // -------------------------------------------------------------------
    int64_t n = static_cast<int64_t>(vector_size);

    if (n > 1) {
        for (int64_t start = (n - 2) / 2; start > 0;
             start = static_cast<int64_t>(
                 std::pow(2, static_cast<int64_t>(
                     std::log2(static_cast<double>(start))))) - 2)
        {
            int64_t end_level = static_cast<int64_t>(
                std::pow(2, static_cast<int64_t>(
                    std::log2(static_cast<double>(start))))) - 2;

            std::size_t items = static_cast<std::size_t>(start - end_level);

            // Mimic original: permanently shrink chunk_size when a level
            // has fewer items than the current chunk_size.
            if (chunk_size > items)
                chunk_size = items / 2;
            if (chunk_size == 0)
                chunk_size = 1;

            // Launch one task per chunk, all with SIMULTANEOUS coherence
            // so they can run in parallel within this level.
            std::vector<Future> futures;
            futures.reserve(items / chunk_size + 1);
            std::size_t cnt = 0;

            while (cnt + chunk_size < items) {
                SiftDownRangeArgs sargs;
                sargs.n            = n;
                sargs.start_offset = start - static_cast<int64_t>(cnt);
                sargs.count        = static_cast<int64_t>(chunk_size);

                TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                    TaskArgument(&sargs, sizeof(SiftDownRangeArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements[0].add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
                cnt += chunk_size;
            }

            // Left-over chunk
            if (cnt < items) {
                SiftDownRangeArgs sargs;
                sargs.n            = n;
                sargs.start_offset = start - static_cast<int64_t>(cnt);
                sargs.count        = static_cast<int64_t>(items - cnt);

                TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                    TaskArgument(&sargs, sizeof(SiftDownRangeArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements[0].add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
            }

            // Required synchronisation per level
            for (auto &f : futures)
                f.get_void_result();
        }

        // Perform sift_down for the head node (root of the heap)
        {
            SiftDownRangeArgs sargs;
            sargs.n            = n;
            sargs.start_offset = 0;
            sargs.count        = 1;

            TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                TaskArgument(&sargs, sizeof(SiftDownRangeArgs)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VAL);

            runtime->execute_task(ctx, launcher).get_void_result();
        }
    }

    // ---- read back results and write heap characteristics -------------
    {
        InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        const int *data = acc.ptr(0);
        write_heap_characteristics(data, vector_size);

        runtime->unmap_region(ctx, pr);
    }

    // ---- clean up -----------------------------------------------------
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// -----------------------------------------------------------------------
// main – register tasks and start the Legion runtime
// -----------------------------------------------------------------------
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
        registrar.set_leaf();
        Runtime::preregister_task_variant<sift_down_range_task>(registrar,
                                                                 "sift_down_range");
    }

    return Runtime::start(argc, argv);
}
