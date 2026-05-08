////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated from HPX/ASTM to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INCREMENT_TASK_ID,
    SQUARE_TASK_ID,
};

enum FieldIDs {
    FID_VALUE,
};

// A = A + 1  (Legion equivalent of increment_transaction)
void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VALUE);
    int val = acc[0];
    acc[0] = val + 1;
}

// A = A * A  (Legion equivalent of square_transaction)
void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_WRITE, int, 1, coord_t,
        Realm::AffineAccessor<int, 1, coord_t>> acc(regions[0], FID_VALUE);
    int val = acc[0];
    acc[0] = val * val;
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open output file
    std::ofstream outfile("concurrency_test.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open concurrency_test.txt" << std::endl;
        return;
    }

    // Create a logical region with a single int element
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VALUE);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // ---------------------------------------------------------
    // TEST 1: High Concurrency Increment
    // Goal: Ensure 64 tasks can atomically increment a shared
    //       variable without data loss.
    //       In Legion, READ_WRITE + EXCLUSIVE on the same region
    //       serializes the tasks, guaranteeing correctness
    //       (equivalent to STM retry-based serialization).
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        // Initialize the shared variable to 0
        runtime->fill_field(ctx, lr, lr, FID_VALUE, 0);

        // Launch 64 async increment tasks
        std::vector<Future> futures;
        for (unsigned i = 0; i < 64; ++i) {
            TaskLauncher launcher(INCREMENT_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }

        // Wait for all to complete
        for (auto &f : futures) {
            f.get_void_result();
        }

        // Read the result via inline mapping
        {
            RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_ONLY, int, 1, coord_t,
                Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VALUE);
            int result = acc[0];

            runtime->unmap_region(ctx, pr);

            outfile << "  Expected Result: 64\n";
            outfile << "  Actual Result:   " << result << "\n";
            outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";
        }
    }

    // ---------------------------------------------------------
    // TEST 2: Exponential Growth (The squaring logic)
    // Goal: Test logic where the write depends heavily on the
    //       read value.
    //       2 -> 4 -> 16 -> 256 -> 65536
    //       READ_WRITE + EXCLUSIVE serializes the four squarings
    //       in launch order, matching STM semantics.
    // ---------------------------------------------------------
    {
        outfile << "Exponential Growth (Squaring) Test\n";
        outfile << "  Initial Value: 2\n";
        outfile << "  Threads: 4\n";
        outfile << "  Operation: A = A * A\n";

        // Initialize the shared variable to 2
        runtime->fill_field(ctx, lr, lr, FID_VALUE, 2);

        // Launch 4 async square tasks
        std::vector<Future> futures;
        for (unsigned i = 0; i < 4; ++i) {
            TaskLauncher launcher(SQUARE_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
            launcher.region_requirements[0].add_field(FID_VALUE);
            futures.push_back(runtime->execute_task(ctx, launcher));
        }

        // Wait for all to complete
        for (auto &f : futures) {
            f.get_void_result();
        }

        // Read the result via inline mapping
        {
            RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();

            const FieldAccessor<READ_ONLY, int, 1, coord_t,
                Realm::AffineAccessor<int, 1, coord_t>> acc(pr, FID_VALUE);
            int result = acc[0];

            runtime->unmap_region(ctx, pr);

            outfile << "  Expected Result: 65536\n";
            outfile << "  Actual Result:   " << result << "\n";
            outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
        }
    }

    // Clean up
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);

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
        registrar.set_leaf();
        Runtime::preregister_task_variant<increment_task>(registrar, "increment");
    }
    {
        TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<square_task>(registrar, "square");
    }

    return Runtime::start(argc, argv);
}
