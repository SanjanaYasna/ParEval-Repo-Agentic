// heaps.cpp — Legion translation of level-parallel make_heap
#include <legion.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

#include "sift.hpp"

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    SIFT_DOWN_RANGE_TASK_ID,
};

enum {
    FID_VAL,
};

// Serialisable argument block for the sift-down-range task
struct SiftDownRangeArgs {
    int64_t n;             // total number of elements
    int64_t start_offset;  // index of 'start' (offset from first)
    int64_t count;         // how many consecutive nodes to sift
};

/* ------------------------------------------------------------------ */
/*  sift_down_range_task                                               */
/*  Leaf task: performs sift_down for a contiguous chunk of nodes.      */
/* ------------------------------------------------------------------ */
void sift_down_range_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    SiftDownRangeArgs args;
    assert(task->arglen == sizeof(args));
    std::memcpy(&args, task->args, sizeof(args));

    // Obtain a raw pointer into the physical instance (dense 1-D region)
    const FieldAccessor<READ_WRITE, int, 1, coord_t,
          Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    int *base = acc.ptr(rect.lo);      // pointer to element 0

    sift_down_range(base, std::less<int>(),
                    static_cast<std::ptrdiff_t>(args.n),
                    static_cast<std::ptrdiff_t>(args.start_offset),
                    static_cast<std::size_t>(args.count));
}

/* ------------------------------------------------------------------ */
/*  top_level_task                                                     */
/*  Orchestrates initialisation, level-parallel heap construction,     */
/*  and result output.                                                 */
/* ------------------------------------------------------------------ */
void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    /* ---------- command-line parsing -------------------------------- */
    std::size_t vector_size = 25;
    std::size_t chunk_size  = 0;

    const InputArgs &cli = Runtime::get_input_args();
    for (int i = 1; i < cli.argc; ++i) {
        if (!std::strcmp(cli.argv[i], "--vector_size") && i + 1 < cli.argc)
            vector_size = static_cast<std::size_t>(std::atoll(cli.argv[++i]));
        else if (!std::strcmp(cli.argv[i], "--chunk_size") && i + 1 < cli.argc)
            chunk_size = static_cast<std::size_t>(std::atoll(cli.argv[++i]));
    }

    if (chunk_size == 0) {
        Machine::ProcessorQuery pq(Machine::get_machine());
        pq.only_kind(Processor::LOC_PROC);
        std::size_t threads = pq.count();
        if (threads == 0) threads = 1;
        chunk_size = vector_size / threads;
        if (chunk_size == 0) chunk_size = 1;
    }

    if (vector_size == 0) return;

    /* ---------- create logical region ------------------------------ */
    Rect<1> bounds(0, static_cast<coord_t>(vector_size - 1));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, bounds);
    FieldSpace     fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    /* ---------- initialise v[i] = i -------------------------------- */
    {
        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, int, 1, coord_t,
              Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);
        for (std::size_t i = 0; i < vector_size; ++i)
            acc[i] = static_cast<int>(i);

        runtime->unmap_region(ctx, pr);
    }

    /* ---------- make_heap  (level-parallel) ------------------------ */
    const int64_t n = static_cast<int64_t>(vector_size);

    if (n > 1) {
        const int64_t last_internal = (n - 2) / 2;

        // Compute the tree level of the last internal node.
        // Node i is at level floor(log2(i+1)).
        // Level k contains nodes [2^k - 1, 2^(k+1) - 2].
        int max_level = 0;
        {
            int64_t tmp = last_internal + 1;
            while (tmp > 1) { tmp >>= 1; max_level++; }
        }

        // Process each tree level from deepest to level 1.
        // Within a level, all subtrees are disjoint so tasks
        // can run concurrently (SIMULTANEOUS coherence).
        for (int level = max_level; level >= 1; --level) {
            int64_t first_at_level = (1LL << level) - 1;
            int64_t last_at_level  = std::min((1LL << (level + 1)) - 2,
                                              last_internal);

            std::size_t items = static_cast<std::size_t>(
                last_at_level - first_at_level + 1);

            std::size_t cs = chunk_size;
            if (cs > items) cs = items / 2;
            if (cs == 0)    cs = 1;

            std::vector<Future> futures;
            futures.reserve(items / cs + 1);

            // Launch one task per chunk – all with SIMULTANEOUS coherence
            // so that they may execute concurrently (their subtrees are
            // disjoint within the same level).
            std::size_t cnt = 0;
            while (cnt + cs < items) {
                SiftDownRangeArgs a;
                a.n            = n;
                a.start_offset = last_at_level - static_cast<int64_t>(cnt);
                a.count        = static_cast<int64_t>(cs);

                TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                                     TaskArgument(&a, sizeof(a)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements.back().add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
                cnt += cs;
            }
            // Left-over (or single-chunk) remainder
            if (cnt < items) {
                SiftDownRangeArgs a;
                a.n            = n;
                a.start_offset = last_at_level - static_cast<int64_t>(cnt);
                a.count        = static_cast<int64_t>(items - cnt);

                TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                                     TaskArgument(&a, sizeof(a)));
                launcher.add_region_requirement(
                    RegionRequirement(lr, READ_WRITE, SIMULTANEOUS, lr));
                launcher.region_requirements.back().add_field(FID_VAL);

                futures.push_back(runtime->execute_task(ctx, launcher));
            }

            // Level barrier – wait for every task at this level before
            // proceeding to the next (higher) level.
            for (auto &f : futures)
                f.get_void_result();
        }

        // Sift down the root (index 0) – single task, EXCLUSIVE coherence.
        {
            SiftDownRangeArgs a;
            a.n            = n;
            a.start_offset = 0;   // root
            a.count        = 1;

            TaskLauncher launcher(SIFT_DOWN_RANGE_TASK_ID,
                                 TaskArgument(&a, sizeof(a)));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements.back().add_field(FID_VAL);

            runtime->execute_task(ctx, launcher).get_void_result();
        }
    }

    /* ---------- write heap characteristics ------------------------- */
    {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<READ_ONLY, int, 1, coord_t,
              Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VAL);

        std::vector<int> v(vector_size);
        for (std::size_t i = 0; i < vector_size; ++i)
            v[i] = acc[i];

        runtime->unmap_region(ctx, pr);

        // Reproduce the exact output format of the original programme
        std::ofstream outFile("heaps.txt");

        std::size_t print_count = std::min(v.size(), std::size_t(10));
        outFile << "First " << print_count << " elements: ";
        for (std::size_t i = 0; i < print_count; ++i) {
            outFile << v[i];
            if (i + 1 < print_count) outFile << " ";
        }
        outFile << "\n";

        outFile << "Last " << print_count << " elements: ";
        for (std::size_t i = v.size() - print_count; i < v.size(); ++i) {
            outFile << v[i];
            if (i + 1 < v.size()) outFile << " ";
        }
        outFile << "\n";

        long long sum = std::accumulate(v.begin(), v.end(), 0LL);
        outFile << "Sum of all elements: " << sum << "\n";

        if (!v.empty())
            outFile << "Root (max) element: " << v[0] << "\n";

        outFile << "Is valid heap: "
                << (std::is_heap(v.begin(), v.end()) ? "true" : "false")
                << "\n";
        outFile.close();
    }

    /* ---------- cleanup -------------------------------------------- */
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

/* ------------------------------------------------------------------ */
/*  main – register tasks and start the Legion runtime                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar r(TOP_LEVEL_TASK_ID, "top_level");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(r, "top_level");
    }
    {
        TaskVariantRegistrar r(SIFT_DOWN_RANGE_TASK_ID, "sift_down_range");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<sift_down_range_task>(
            r, "sift_down_range");
    }

    return Runtime::start(argc, argv);
}
