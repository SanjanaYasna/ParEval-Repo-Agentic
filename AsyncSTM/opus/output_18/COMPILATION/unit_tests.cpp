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
#include <cassert>

using namespace Legion;

enum FieldIDs {
    FID_VAL = 0,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    CHECK_VALUE_TASK_ID,
};

// ---------------------------------------------------------------------------
// Helper: create a logical region holding a single int, initialized to init_val
// ---------------------------------------------------------------------------
LogicalRegion create_int_var(Context ctx, Runtime *runtime, int init_val)
{
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = init_val;
    runtime->unmap_region(ctx, pr);

    return lr;
}

// ---------------------------------------------------------------------------
// Helper: read the single int value from a region
// ---------------------------------------------------------------------------
int read_int_var(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
    int val = acc[0];
    runtime->unmap_region(ctx, pr);
    return val;
}

// ---------------------------------------------------------------------------
// Helper: write a single int value into a region
// ---------------------------------------------------------------------------
void write_int_var(Context ctx, Runtime *runtime, LogicalRegion lr, int val)
{
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = val;
    runtime->unmap_region(ctx, pr);
}

// ---------------------------------------------------------------------------
// Helper: destroy a logical region and its index/field spaces
// ---------------------------------------------------------------------------
void destroy_var(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    IndexSpace is = lr.get_index_space();
    FieldSpace fs = lr.get_field_space();
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// check_value_task: receives an int via TaskArgument, asserts it equals 16.
// Equivalent to the IO.then() lambda in the original ASTM code.
// ---------------------------------------------------------------------------
void check_value_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int));
    int val = *(const int *)task->args;
    assert(val == 16);
}

// ---------------------------------------------------------------------------
// top_level_task – runs all unit tests sequentially
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    { // Read A, Write A
        // Original: shared_var<int> A(1); transaction { A_ = 2; } → A = 2
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        // Transaction equivalent: read A, then write A = 2
        {
            RegionRequirement req(A_lr, READ_WRITE, EXCLUSIVE, A_lr);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            // implicit read of acc[0] (= 1), then write
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Write A, Read A
        // Original: shared_var<int> A(1); transaction { A_ = 2; } → A = 2
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        {
            RegionRequirement req(A_lr, READ_WRITE, EXCLUSIVE, A_lr);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Read A): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Write A, Write A  (actually reads A and asserts twice)
        // Original: shared_var<int> A(1); transaction { assert(A_==1); assert(A_==1); }
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        {
            int val = read_int_var(ctx, runtime, A_lr);
            assert(val == 1);
            assert(val == 1);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Write A, Write A (overwrite)
        // Original: shared_var<int> A(1); transaction { A_ = 2; A_ = 2; } → A = 2
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        write_int_var(ctx, runtime, A_lr, 2);
        write_int_var(ctx, runtime, A_lr, 2);

        outfile << "Test (Overwrite): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Read A, Write A, Read A
        // Original: shared_var<int> A(1); transaction { A_ = 2; } → A = 2
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        write_int_var(ctx, runtime, A_lr, 2);

        outfile << "Test (Read -> Write -> Read): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Write A, Read A, Write A
        // Original: shared_var<int> A(1); transaction { A_ = 2; A_ = 2; } → A = 2
        LogicalRegion A_lr = create_int_var(ctx, runtime, 1);

        write_int_var(ctx, runtime, A_lr, 2);
        write_int_var(ctx, runtime, A_lr, 2);

        outfile << "Test (Write -> Read -> Write): A = "
                << read_int_var(ctx, runtime, A_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    { // Basic arithmetic with local_vars
        // Original: A=4, B=1; transaction { A_ = A_*A_ - B_; } → A = 15, B = 1
        LogicalRegion A_lr = create_int_var(ctx, runtime, 4);
        LogicalRegion B_lr = create_int_var(ctx, runtime, 1);

        {
            int A_val = read_int_var(ctx, runtime, A_lr);
            int B_val = read_int_var(ctx, runtime, B_lr);
            write_int_var(ctx, runtime, A_lr, A_val * A_val - B_val);
        }

        outfile << "Test (Arithmetic): A = " << read_int_var(ctx, runtime, A_lr)
                << ", B = " << read_int_var(ctx, runtime, B_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
        destroy_var(ctx, runtime, B_lr);
    }

    { // Read A to future
        // Original: A=4, B=1; transaction { A_=A_*A_; IO.then(check A==16);
        //           A_=A_-B_; }  IO.get(); → A = 15, B = 1
        LogicalRegion A_lr = create_int_var(ctx, runtime, 4);
        LogicalRegion B_lr = create_int_var(ctx, runtime, 1);

        int A_val = read_int_var(ctx, runtime, A_lr);
        int B_val = read_int_var(ctx, runtime, B_lr);

        A_val = A_val * A_val;   // A intermediate = 16
        int local_A = A_val;     // Capture intermediate for async check

        // Launch async check task (equivalent to IO.then)
        TaskLauncher check_launcher(CHECK_VALUE_TASK_ID,
                                    TaskArgument(&local_A, sizeof(local_A)));
        Future f = runtime->execute_task(ctx, check_launcher);

        A_val = A_val - B_val;   // A final = 15
        write_int_var(ctx, runtime, A_lr, A_val);

        // Wait for async check to complete (equivalent to IO.get())
        f.get_void_result();

        outfile << "Test (Read A to Future): A = "
                << read_int_var(ctx, runtime, A_lr)
                << ", B = " << read_int_var(ctx, runtime, B_lr) << std::endl;
        destroy_var(ctx, runtime, A_lr);
        destroy_var(ctx, runtime, B_lr);
    }

    { // Retry Logic
        // Original: A=4; transaction attempt 1: read A=4, tmp=16, external
        // write A=3, commit fails (read mismatch). Attempt 2: read A=3,
        // tmp=9, commit succeeds. → A = 9, Attempts = 2.
        //
        // Legion has no optimistic-concurrency retry, so we model the
        // validation-based retry explicitly: snapshot the value, perform
        // the computation, re-read to validate, and retry on mismatch.
        LogicalRegion A_lr = create_int_var(ctx, runtime, 4);

        bool fail = true;
        int attempt_count = 0;
        bool committed = false;

        while (!committed) {
            ++attempt_count;

            // Snapshot the current value (equivalent to transactional read)
            int A_val = read_int_var(ctx, runtime, A_lr);
            int snapshot = A_val;
            int tmp = A_val * A_val;

            if (fail) {
                // Simulate an external concurrent write that causes conflict
                write_int_var(ctx, runtime, A_lr, 3);
                fail = false;
            }

            // Validate: re-read and compare against snapshot
            int current = read_int_var(ctx, runtime, A_lr);
            if (current == snapshot) {
                // No conflict detected – commit the write
                write_int_var(ctx, runtime, A_lr, tmp);
                committed = true;
            }
            // else: conflict detected, loop retries
        }

        outfile << "Test (Retry Logic): A = "
                << read_int_var(ctx, runtime, A_lr)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_var(ctx, runtime, A_lr);
    }

    outfile.close();
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
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
        TaskVariantRegistrar registrar(CHECK_VALUE_TASK_ID, "check_value");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<check_value_task>(registrar, "check_value");
    }

    return Runtime::start(argc, argv);
}
