#include "legion.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cstdlib>

#include "sift.hpp"

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    SIFT_CHUNK_TASK_ID,
    SIFT_ROOT_TASK_ID,
};

enum {
    FID_VAL,
};

// Arguments passed to each sift-down chunk task
struct ChunkArgs {
    long long n;              // total number of elements
    long long start_offset;   // index of the first node to sift in this chunk
    long long count;          // how many consecutive nodes to sift
};

// ---------------------------------------------------------------------------
// Sift-chunk task: calls sift_down_range from sift.hpp using raw-pointer
// iterators obtained from the Legion physical region.
// ---------------------------------------------------------------------------
void sift_chunk_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    ChunkArgs args;
    memcpy(&args, task->args, sizeof(ChunkArgs));

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
          Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);
    int *first = acc.ptr(0);

    // Call sift_down_range with pointer+index based signature
    sift_down_range<int>(first, std::less<int>(),
                         static_cast<int64_t>(args.n),
                         static_cast<int64_t>(args.start_offset),
                         static_cast<int64_t>(args.count));
}

// ---------------------------------------------------------------------------
// Sift-root task: sift_down the root (index 0) after all other levels
// ---------------------------------------------------------------------------
void sift_root_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    long long n;
    memcpy(&n, task->args, sizeof(long long));

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
          Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);
    int *first = acc.ptr(0);

    sift_down<int>(first, std::less<int>(), static_cast<int64_t>(n), 0);
}

// ---------------------------------------------------------------------------
// write_heap_characteristics – identical semantics to the HPX version
// ---------------------------------------------------------------------------
void write_heap_characteristics(const std::vector<int>& v)
{
    std::ofstream outFile("heaps.txt");
    std::size_t sz = v.size();
    std::size_t print_count = std::min(sz, std::size_t(10));

    outFile << "First " << print_count << " elements: ";
    for (std::size_t i = 0; i < print_count; ++i) {
        outFile << v[i];
        if (i < print_count - 1) outFile << " ";
    }
    outFile << "\n";

    outFile << "Last " << print_count << " elements: ";
    std::size_t last_start = (sz >= 10) ? sz - 10 : 0;
    for (std::size_t i = last_start; i < sz; ++i) {
        outFile << v[i];
        if (i < sz - 1) outFile << " ";
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

// ---------------------------------------------------------------------------
// Top-level task – orchestrates initialization, level-parallel make_heap,
// and result output.
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- Parse command-line arguments ----
    long long vector_size = 25;
    long long chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--vector_size") &&
            i + 1 < command_args.argc) {
            vector_size = atoll(command_args.argv[++i]);
        } else if (!strcmp(command_args.argv[i], "--chunk_size") &&
                   i + 1 < command_args.argc) {
            chunk_size = atoll(command_args.argv[++i]);
        }
    }

    // Auto-select chunk_size based on available CPU processors (mirrors
    // the HPX code that uses hardware_concurrency).
    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        long long num_procs = static_cast<long long>(pq.count());
        chunk_size = (num_procs > 0) ? vector_size / num_procs : vector_size;
        if (chunk_size < 1) chunk_size = 1;
    }

    // ---- Create logical region for the vector ----
    IndexSpace is = runtime->create_index_space(ctx,
                        Rect<1>(0, vector_size - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---- Initialise with iota: 0, 1, 2, … , vector_size-1 ----
    {
        InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1, coord_t,
              Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        int *data = acc.ptr(0);
        for (long long i = 0; i < vector_size; i++)
            data[i] = static_cast<int>(i);

        runtime->unmap_region(ctx, pr);
    }

    // ---- make_heap: bottom-up, level-parallel construction ----
    //
    // Outer loop iterates over levels (right-most element of each level).
    // Inner loop partitions each level into chunks and launches them as
    // concurrent Legion tasks with SIMULTANEOUS coherence (the subtrees
    // rooted at nodes within the same level are disjoint, so no data
    // races occur).  Between levels we wait on all futures – this is the
    // same barrier semantics as hpx::wait_all in the original code.
    //
    long long n = vector_size;
    if (n > 1) {
        for (long long start = (n - 2) / 2; start > 0;
             start = static_cast<long long>(
                         std::pow(2.0, static_cast<long long>(
                                           std::log2(static_cast<double>(start))))) - 2)
        {
            long long end_level = static_cast<long long>(
                std::pow(2.0, static_cast<long long>(
                                  std::log2(static_cast<double>(start))))) - 2;
            long long items = start - end_level;

            long long cs = chunk_size;
            if (cs > items)
                cs = items / 2;
            if (cs < 1)
                cs = 1;

            std::vector<Future> futures;

            long long cnt = 0;
            while (cnt + cs < items) {
                ChunkArgs args;
                args.n            = n;
                args.start_offset = start - cnt;
                args.count        = cs;

                TaskLauncher launcher(SIFT_CHUNK_TASK_ID,
                    TaskArgument(&args, sizeof(ChunkArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements[0].add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
                cnt += cs;
            }
            if (cnt < items) {
                ChunkArgs args;
                args.n            = n;
                args.start_offset = start - cnt;
                args.count        = items - cnt;

                TaskLauncher launcher(SIFT_CHUNK_TASK_ID,
                    TaskArgument(&args, sizeof(ChunkArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements[0].add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
            }

            // Level barrier: wait for every chunk at this level
            for (auto &f : futures)
                f.get_void_result();
        }

        // Sift down the root node (single task, exclusive access)
        {
            long long n_val = n;
            TaskLauncher launcher(SIFT_ROOT_TASK_ID,
                TaskArgument(&n_val, sizeof(long long)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VAL);

            runtime->execute_task(ctx, launcher).get_void_result();
        }
    }

    // ---- Read back the heap and write characteristics to heaps.txt ----
    {
        InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
              Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        const int *data = acc.ptr(0);

        std::vector<int> v(static_cast<std::size_t>(vector_size));
        for (long long i = 0; i < vector_size; i++)
            v[static_cast<std::size_t>(i)] = data[i];

        runtime->unmap_region(ctx, pr);

        write_heap_characteristics(v);
    }

    // ---- Cleanup ----
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// main – register all tasks and start the Legion runtime
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(SIFT_CHUNK_TASK_ID, "sift_chunk");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<sift_chunk_task>(registrar, "sift_chunk");
    }
    {
        TaskVariantRegistrar registrar(SIFT_ROOT_TASK_ID, "sift_root");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<sift_root_task>(registrar, "sift_root");
    }

    return Runtime::start(argc, argv);
}
