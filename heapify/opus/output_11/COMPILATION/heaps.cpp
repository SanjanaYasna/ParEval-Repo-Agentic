// heaps.cpp – Legion translation of the HPX level-parallel make_heap
#include "legion.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cassert>
#include <cstdint>

#include "sift.hpp"

using namespace Legion;

// ---- Task and Field IDs ----
enum {
    TOP_LEVEL_TASK_ID,
    SIFT_DOWN_RANGE_TASK_ID,
    SIFT_DOWN_ROOT_TASK_ID,
};

enum {
    FID_VAL = 0,
};

// Per-chunk argument passed through the ArgumentMap of an index launch
struct SiftArgs {
    int64_t n;          // total number of elements
    int64_t start_pos;  // starting heap index for this chunk
    int64_t count;      // how many nodes this chunk sifts down
};

// ---------- leaf task: sift-down a range of nodes ----------
void sift_down_range_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(task->local_arglen == sizeof(SiftArgs));
    SiftArgs args = *reinterpret_cast<const SiftArgs *>(task->local_args);

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    int *first = acc.ptr(Point<1>(0));

    sift_down_range(first, args.n, args.start_pos,
                    args.count);
}

// ---------- leaf task: sift-down the root only ----------
void sift_down_root_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int64_t));
    int64_t n = *reinterpret_cast<const int64_t *>(task->args);

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    int *first = acc.ptr(Point<1>(0));

    sift_down(first, n, 0);
}

// ---------- output (kept identical to the HPX version) ----------
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
    for (std::size_t i = v.size() - print_count; i < v.size(); ++i) {
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

// ---------- top-level task ----------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- Parse command-line arguments ----
    int64_t vector_size = 25;
    int64_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--vector_size") == 0 &&
            i + 1 < command_args.argc) {
            vector_size = atoll(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--chunk_size") == 0 &&
                   i + 1 < command_args.argc) {
            chunk_size = atoll(command_args.argv[++i]);
        }
    }

    // Auto-select chunk_size from available CPU processors
    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        std::size_t num_procs = pq.count();
        if (num_procs == 0) num_procs = 1;
        chunk_size = vector_size / static_cast<int64_t>(num_procs);
        if (chunk_size <= 0) chunk_size = 1;
    }

    // ---- Create logical region for the array ----
    IndexSpace is = runtime->create_index_space(ctx,
                        Rect<1>(0, vector_size - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---- Initialise with iota (0, 1, 2, …) ----
    {
        InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);

        for (int64_t i = 0; i < vector_size; i++)
            acc[i] = static_cast<int>(i);

        runtime->unmap_region(ctx, pr);
    }

    // ---- Level-parallel make_heap ----
    int64_t n = vector_size;

    if (n > 1) {
        // Walk from the last internal node up to the root, one tree level
        // at a time.  Within each level the sift-downs are independent, so
        // they are launched as an index-space task with SIMULTANEOUS
        // coherence (disjoint element accesses, same physical instance).
        for (int64_t start = (n - 2) / 2; start > 0;
             start = static_cast<int64_t>(
                         pow(2.0, static_cast<int64_t>(
                                      log2(static_cast<double>(start))))) - 2)
        {
            int64_t end_level = static_cast<int64_t>(
                pow(2.0, static_cast<int64_t>(
                             log2(static_cast<double>(start))))) - 2;
            int64_t items = start - end_level;

            int64_t cs = chunk_size;
            if (cs > items)
                cs = items / 2;
            if (cs <= 0) cs = 1;

            // Count how many chunks we will launch
            int num_chunks = 0;
            {
                int64_t cnt = 0;
                while (cnt + cs < items) { num_chunks++; cnt += cs; }
                if (cnt < items) num_chunks++;
            }
            if (num_chunks <= 0) continue;

            // Build per-point arguments
            ArgumentMap arg_map;
            {
                int64_t cnt = 0;
                int idx = 0;
                while (cnt + cs < items) {
                    SiftArgs sa;
                    sa.n         = n;
                    sa.start_pos = start - cnt;
                    sa.count     = cs;
                    arg_map.set_point(DomainPoint(Point<1>(idx)),
                                      TaskArgument(&sa, sizeof(SiftArgs)));
                    idx++;
                    cnt += cs;
                }
                if (cnt < items) {
                    SiftArgs sa;
                    sa.n         = n;
                    sa.start_pos = start - cnt;
                    sa.count     = items - cnt;
                    arg_map.set_point(DomainPoint(Point<1>(idx)),
                                      TaskArgument(&sa, sizeof(SiftArgs)));
                }
            }

            // Index launch – all chunks for this level execute concurrently
            IndexSpace launch_is = runtime->create_index_space(ctx,
                                       Rect<1>(0, num_chunks - 1));

            IndexLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                                   launch_is,
                                   TaskArgument(NULL, 0),
                                   arg_map);
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
            launcher.region_requirements[0].add_field(FID_VAL);

            // Blocking: acts as the per-level barrier
            FutureMap fm = runtime->execute_index_space(ctx, launcher);
            fm.wait_all_results();

            runtime->destroy_index_space(ctx, launch_is);
        }

        // Sift down the root node (single task)
        {
            int64_t n_val = n;
            TaskLauncher launcher(SIFT_DOWN_ROOT_TASK_ID,
                                  TaskArgument(&n_val, sizeof(int64_t)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }

    // ---- Read back the heap and write diagnostics ----
    {
        InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);

        std::vector<int> v(vector_size);
        for (int64_t i = 0; i < vector_size; i++)
            v[i] = acc[i];

        runtime->unmap_region(ctx, pr);
        write_heap_characteristics(v);
    }

    // ---- Cleanup ----
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------- registration & entry point ----------
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
    {
        TaskVariantRegistrar registrar(SIFT_DOWN_ROOT_TASK_ID, "sift_down_root");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<sift_down_root_task>(registrar,
                                                                "sift_down_root");
    }

    return Runtime::start(argc, argv);
}
