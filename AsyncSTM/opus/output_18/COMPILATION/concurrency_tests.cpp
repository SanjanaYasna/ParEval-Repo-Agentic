////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <iostream>
#include <fstream>

using namespace Legion;

enum {
    TOP_LEVEL_TASK_ID,
    INCREMENT_TASK_ID,
    SQUARE_TASK_ID,
};

enum {
    FID_VALUE,
};

// A = A + 1
// Legion serializes tasks with READ_WRITE on the same region,
// replacing the STM commit/retry loop.
void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    int val = acc[0];
    acc[0] = val + 1;
}

// A = A * A
void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    int val = acc[0];
    acc[0] = val * val;
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open output file
    std::ofstream outfile("concurrency_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
        return;
    }

    // ---------------------------------------------------------
    // TEST 1: High Concurrency Increment
    // Goal: Ensure 64 tasks can atomically increment a shared
    //       variable without data loss.
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        // Create a single-element logical region for shared_var<int> A(0)
        Rect<1> rect(0, 0);
        IndexSpace is = runtime->create_index_space(ctx, rect);
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(int), FID_VALUE);
        }
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Initialize A to 0
        runtime->fill_field(ctx, lr, lr, FID_VALUE, 0);

        // Launch 64 async increment tasks (analogous to ASTM_ASYNC).
        // All tasks require READ_WRITE on the same region, so the
        // Legion runtime serializes them — exactly like the STM's
        // optimistic-retry mechanism, but without retries.
        for (unsigned i = 0; i < 64; ++i) {
            TaskLauncher launcher(INCREMENT_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VALUE);
            runtime->execute_task(ctx, launcher);
        }

        // Read the result after all increment tasks have completed.
        // The inline mapping implicitly waits for all prior writers.
        RegionRequirement read_req(lr, READ_ONLY, EXCLUSIVE, lr);
        read_req.add_field(FID_VALUE);
        InlineLauncher il(read_req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
        int result = acc[0];
        runtime->unmap_region(ctx, pr);

        outfile << "  Expected Result: 64\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";

        // Clean up
        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    // ---------------------------------------------------------
    // TEST 2: Exponential Growth (The 'cubeA' logic)
    // Goal: Test logic where the write depends heavily on the
    //       read value.
    // We start at 2 and run 4 iterations:
    //   2 -> 4 -> 16 -> 256 -> 65536
    // ---------------------------------------------------------
    {
        outfile << "Exponential Growth (Squaring) Test\n";
        outfile << "  Initial Value: 2\n";
        outfile << "  Threads: 4\n";
        outfile << "  Operation: A = A * A\n";

        // Create a single-element logical region for shared_var<int> A(2)
        Rect<1> rect(0, 0);
        IndexSpace is = runtime->create_index_space(ctx, rect);
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(int), FID_VALUE);
        }
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Initialize A to 2
        runtime->fill_field(ctx, lr, lr, FID_VALUE, 2);

        // Launch 4 async squaring tasks.
        // READ_WRITE serializes them in launch order:
        //   2 -> 4 -> 16 -> 256 -> 65536
        for (unsigned i = 0; i < 4; ++i) {
            TaskLauncher launcher(SQUARE_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VALUE);
            runtime->execute_task(ctx, launcher);
        }

        // Read the result
        RegionRequirement read_req(lr, READ_ONLY, EXCLUSIVE, lr);
        read_req.add_field(FID_VALUE);
        InlineLauncher il(read_req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
        int result = acc[0];
        runtime->unmap_region(ctx, pr);

        outfile << "  Expected Result: 65536\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";

        // Clean up
        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    outfile.close();
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
        TaskVariantRegistrar registrar(INCREMENT_TASK_ID, "increment");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<increment_task>(registrar, "increment");
    }
    {
        TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<square_task>(registrar, "square");
    }

    return Runtime::start(argc, argv);
}
