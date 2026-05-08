////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "legion.h"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INCREMENT_TASK_ID,
    SQUARE_TASK_ID,
    READ_TASK_ID,
};

enum FieldIDs {
    FID_VALUE,
};

// A = A + 1  (equivalent to increment_transaction)
void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    Point<1> p(0);
    int val = acc[p];
    acc[p] = val + 1;
}

// A = A * A  (equivalent to square_transaction)
void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    Point<1> p(0);
    int val = acc[p];
    acc[p] = val * val;
}

// Read-only task that returns the current value
int read_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, int, 1> acc(regions[0], FID_VALUE);
    Point<1> p(0);
    return acc[p];
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
    // Goal: Ensure 64 tasks atomically increment a shared variable
    //       without data loss. Legion serializes all READ_WRITE
    //       accesses to the same region, equivalent to STM retry.
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        // Create a logical region with a single element
        IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator allocator =
                runtime->create_field_allocator(ctx, fs);
            allocator.allocate_field(sizeof(int), FID_VALUE);
        }
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Initialize shared variable to 0
        {
            RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
            acc[Point<1>(0)] = 0;
            runtime->unmap_region(ctx, pr);
        }

        // Launch 64 increment tasks.
        // All have READ_WRITE on the same region so Legion
        // chains them, guaranteeing serialized execution (atomicity).
        for (unsigned i = 0; i < 64; ++i) {
            TaskLauncher launcher(INCREMENT_TASK_ID,
                                  TaskArgument(NULL, 0));
            RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            launcher.add_region_requirement(req);
            runtime->execute_task(ctx, launcher);
        }

        // Read the result — READ_ONLY is ordered after all prior
        // READ_WRITE tasks on the same region.
        int result = 0;
        {
            TaskLauncher launcher(READ_TASK_ID,
                                  TaskArgument(NULL, 0));
            RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            launcher.add_region_requirement(req);
            Future f = runtime->execute_task(ctx, launcher);
            result = f.get_result<int>();
        }

        outfile << "  Expected Result: 64\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: "
                << (result == 64 ? "PASS" : "FAIL") << "\n\n";

        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    // ---------------------------------------------------------
    // TEST 2: Exponential Growth (Squaring)
    // Goal: Test logic where the write depends heavily on the
    //       read value.
    //   2 -> 4 -> 16 -> 256 -> 65536
    // ---------------------------------------------------------
    {
        outfile << "Exponential Growth (Squaring) Test\n";
        outfile << "  Initial Value: 2\n";
        outfile << "  Threads: 4\n";
        outfile << "  Operation: A = A * A\n";

        IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
        FieldSpace fs = runtime->create_field_space(ctx);
        {
            FieldAllocator allocator =
                runtime->create_field_allocator(ctx, fs);
            allocator.allocate_field(sizeof(int), FID_VALUE);
        }
        LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

        // Initialize shared variable to 2
        {
            RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
            acc[Point<1>(0)] = 2;
            runtime->unmap_region(ctx, pr);
        }

        // Launch 4 squaring tasks — serialized by READ_WRITE dependence
        for (unsigned i = 0; i < 4; ++i) {
            TaskLauncher launcher(SQUARE_TASK_ID,
                                  TaskArgument(NULL, 0));
            RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            launcher.add_region_requirement(req);
            runtime->execute_task(ctx, launcher);
        }

        // Read result
        int result = 0;
        {
            TaskLauncher launcher(READ_TASK_ID,
                                  TaskArgument(NULL, 0));
            RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            launcher.add_region_requirement(req);
            Future f = runtime->execute_task(ctx, launcher);
            result = f.get_result<int>();
        }

        outfile << "  Expected Result: 65536\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: "
                << (result == 65536 ? "PASS" : "FAIL") << "\n";

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
        Runtime::preregister_task_variant<top_level_task>(
            registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(INCREMENT_TASK_ID, "increment");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<increment_task>(
            registrar, "increment");
    }
    {
        TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<square_task>(
            registrar, "square");
    }
    {
        TaskVariantRegistrar registrar(READ_TASK_ID, "read");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<int, read_task>(
            registrar, "read");
    }

    return Runtime::start(argc, argv);
}
