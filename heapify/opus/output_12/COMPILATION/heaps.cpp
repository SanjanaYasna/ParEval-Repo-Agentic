//source: https://github.com/Syntaf/heapify/tree/master (translated to Legion)
#include "legion.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <cstring>
#include <functional>

#include "sift.hpp"

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    SIFT_DOWN_CHUNK_TASK_ID,
};

enum {
    FID_VAL,
};

struct ChunkArgs {
    int64_t total_n;
    int64_t start_pos;
    int64_t count;
};

void sift_down_chunk_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    ChunkArgs args;
    if (task->local_arglen >= sizeof(ChunkArgs))
        memcpy(&args, task->local_args, sizeof(ChunkArgs));
    else
        memcpy(&args, task->args, sizeof(ChunkArgs));

    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    int *first = acc.ptr(Point<1>(0));

    sift_down_range(first, static_cast<long long>(args.total_n),
                    static_cast<long long>(args.start_pos),
                    static_cast<std::size_t>(args.count),
                    std::less<int>());
}

void write_heap_characteristics(const std::vector<int>& v)
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
    int64_t vector_size = 25;
    int64_t chunk_size  = 0;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--vector_size") && i + 1 < command_args.argc) {
            vector_size = atoll(command_args.argv[++i]);
        } else if (!strcmp(command_args.argv[i], "--chunk_size") && i + 1 < command_args.argc) {
            chunk_size = atoll(command_args.argv[++i]);
        }
    }

    if (chunk_size == 0) {
        Machine machine = Machine::get_machine();
        Machine::ProcessorQuery pq(machine);
        pq.only_kind(Processor::LOC_PROC);
        std::size_t num_procs = pq.count();
        if (num_procs == 0) num_procs = 1;
        chunk_size = vector_size / static_cast<int64_t>(num_procs);
        if (chunk_size == 0) chunk_size = 1;
    }

    // Create index space, field space, and logical region for the vector
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, vector_size - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Initialize the vector with 0, 1, 2, ..., vector_size-1 via inline mapping
    {
        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
        for (int64_t i = 0; i < vector_size; i++)
            acc[i] = static_cast<int>(i);
        runtime->unmap_region(ctx, pr);
    }

    // ---- make_heap: bottom-up, level-parallel construction ----
    int64_t n = vector_size;
    if (n > 1) {
        typedef int64_t difference_type;

        for (difference_type start = (n - 2) / 2; start > 0;
             start = static_cast<difference_type>(
                 pow(2.0, static_cast<double>(
                     static_cast<difference_type>(log2(static_cast<double>(start)))))) - 2)
        {
            difference_type end_level = static_cast<difference_type>(
                pow(2.0, static_cast<double>(
                    static_cast<difference_type>(log2(static_cast<double>(start)))))) - 2;

            std::size_t items = static_cast<std::size_t>(start - end_level);
            std::size_t cs = static_cast<std::size_t>(chunk_size);

            // If the chunk_size has become too big, split the level into two chunks
            if (cs > items)
                cs = items / 2;
            if (cs == 0)
                cs = 1;

            // Build per-chunk descriptors
            std::vector<ChunkArgs> chunks;
            std::size_t cnt = 0;
            while (cnt + cs < items) {
                ChunkArgs a;
                a.total_n   = n;
                a.start_pos = start - static_cast<int64_t>(cnt);
                a.count     = static_cast<int64_t>(cs);
                chunks.push_back(a);
                cnt += cs;
            }
            if (cnt < items) {
                ChunkArgs a;
                a.total_n   = n;
                a.start_pos = start - static_cast<int64_t>(cnt);
                a.count     = static_cast<int64_t>(items - cnt);
                chunks.push_back(a);
            }

            if (chunks.size() == 1) {
                // Single task — use EXCLUSIVE coherence
                TaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID,
                    TaskArgument(&chunks[0], sizeof(ChunkArgs)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                runtime->execute_task(ctx, launcher).get_void_result();
            } else {
                // Index launch — SIMULTANEOUS coherence allows parallel
                // execution on the same physical instance (subtrees are disjoint)
                Rect<1> launch_bounds(0, static_cast<coord_t>(chunks.size() - 1));
                IndexSpace launch_is = runtime->create_index_space(ctx, launch_bounds);

                ArgumentMap arg_map;
                for (std::size_t i = 0; i < chunks.size(); i++) {
                    arg_map.set_point(Point<1>(static_cast<coord_t>(i)),
                        TaskArgument(&chunks[i], sizeof(ChunkArgs)));
                }

                IndexTaskLauncher idx_launcher(SIFT_DOWN_CHUNK_TASK_ID,
                    launch_is, TaskArgument(NULL, 0), arg_map);
                idx_launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                idx_launcher.region_requirements[0].add_field(FID_VAL);

                // Required synchronization per level
                FutureMap fm = runtime->execute_index_space(ctx, idx_launcher);
                fm.wait_all_results();

                runtime->destroy_index_space(ctx, launch_is);
            }
        }

        // Perform sift down for the head node
        ChunkArgs root_args;
        root_args.total_n   = n;
        root_args.start_pos = 0;
        root_args.count     = 1;
        TaskLauncher launcher(SIFT_DOWN_CHUNK_TASK_ID,
            TaskArgument(&root_args, sizeof(ChunkArgs)));
        launcher.add_region_requirement(
            RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher).get_void_result();
    }

    // Read back the results and write heap characteristics
    {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
        std::vector<int> v(static_cast<std::size_t>(vector_size));
        for (int64_t i = 0; i < vector_size; i++)
            v[static_cast<std::size_t>(i)] = acc[i];
        runtime->unmap_region(ctx, pr);

        write_heap_characteristics(v);
    }

    // Clean up
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
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(SIFT_DOWN_CHUNK_TASK_ID, "sift_down_chunk");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<sift_down_chunk_task>(registrar, "sift_down_chunk");
    }

    return Runtime::start(argc, argv);
}
