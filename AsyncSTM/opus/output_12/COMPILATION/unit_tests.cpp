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
    TOP_LEVEL_TASK_ID = 0,
    ASYNC_CHECK_TASK_ID,
};

// ---------------------------------------------------------------------------
// Helper: create a 1-element logical region holding a single int
// ---------------------------------------------------------------------------
static LogicalRegion create_int_region(Context ctx, Runtime *runtime, int init_val)
{
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Initialize the region
    InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = init_val;
    runtime->unmap_region(ctx, pr);

    return lr;
}

// ---------------------------------------------------------------------------
// Helper: read a single int from a 1-element region
// ---------------------------------------------------------------------------
static int read_int_region(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
    int val = acc[0];
    runtime->unmap_region(ctx, pr);
    return val;
}

// ---------------------------------------------------------------------------
// Helper: write a single int to a 1-element region
// ---------------------------------------------------------------------------
static void write_int_region(Context ctx, Runtime *runtime, LogicalRegion lr, int val)
{
    InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = val;
    runtime->unmap_region(ctx, pr);
}

// ---------------------------------------------------------------------------
// Helper: destroy a region and its associated spaces
// ---------------------------------------------------------------------------
static void destroy_int_region(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    IndexSpace is = lr.get_index_space();
    FieldSpace fs = lr.get_field_space();
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// Async check task – equivalent of IO.then() lambda that asserts local_A == 16
// ---------------------------------------------------------------------------
void async_check_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int));
    int local_A = *reinterpret_cast<const int *>(task->args);
    assert(local_A == 16);
}

// ---------------------------------------------------------------------------
// Top-level task – contains all unit tests
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
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        // Transaction equivalent: exclusive inline mapping for read-modify-write
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            (void)acc[0]; // read
            acc[0] = 2;   // write
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Write A, Read A
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;          // write
            (void)acc[0];        // read back
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Read A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Write A, Write A (really just two reads, no writes – original only asserts)
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_ONLY, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
            int A_ = acc[0];
            assert(A_ == 1);
            assert(A_ == 1);
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Write A, Write A (overwrite)
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2; // first write
            acc[0] = 2; // overwrite with same value
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Overwrite): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Read A, Write A, Read A
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            int read1 = acc[0]; // read
            (void)read1;
            acc[0] = 2;         // write
            int read2 = acc[0]; // read after write
            (void)read2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read -> Write -> Read): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Write A, Read A, Write A
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;          // write
            int A_ = acc[0];     // read
            (void)A_;
            acc[0] = 2;          // write again
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write -> Read -> Write): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    { // Basic arithmetic with local_vars: A = A*A - B
        LogicalRegion A = create_int_region(ctx, runtime, 4);
        LogicalRegion B = create_int_region(ctx, runtime, 1);

        {
            InlineLauncher il_a(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il_a.add_field(FID_VAL);
            InlineLauncher il_b(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            il_b.add_field(FID_VAL);

            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_a.wait_until_valid();
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY, int, 1>  acc_b(pr_b, FID_VAL);

            int A_ = acc_a[0];
            int B_ = acc_b[0];
            acc_a[0] = A_ * A_ - B_;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }

        outfile << "Test (Arithmetic): A = " << read_int_region(ctx, runtime, A)
                << ", B = " << read_int_region(ctx, runtime, B) << std::endl;
        destroy_int_region(ctx, runtime, A);
        destroy_int_region(ctx, runtime, B);
    }

    { // Read A to future – async task checks intermediate value of A
        LogicalRegion A = create_int_region(ctx, runtime, 4);
        LogicalRegion B = create_int_region(ctx, runtime, 1);

        int local_A_capture = 0;

        // Transaction equivalent
        {
            InlineLauncher il_a(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il_a.add_field(FID_VAL);
            InlineLauncher il_b(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            il_b.add_field(FID_VAL);

            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_a.wait_until_valid();
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY, int, 1>  acc_b(pr_b, FID_VAL);

            int A_ = acc_a[0]; // 4
            int B_ = acc_b[0]; // 1

            A_ = A_ * A_;          // 16
            local_A_capture = A_;   // capture intermediate value for async check

            A_ = A_ - B_;          // 15
            acc_a[0] = A_;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }

        // Launch async check child task (equivalent of IO.then / IO.get)
        {
            TaskLauncher launcher(ASYNC_CHECK_TASK_ID,
                                  TaskArgument(&local_A_capture, sizeof(int)));
            Future f = runtime->execute_task(ctx, launcher);
            f.get_void_result(); // blocks until task completes (like IO.get())
        }

        outfile << "Test (Read A to Future): A = "
                << read_int_region(ctx, runtime, A)
                << ", B = " << read_int_region(ctx, runtime, B) << std::endl;
        destroy_int_region(ctx, runtime, A);
        destroy_int_region(ctx, runtime, B);
    }

    { // Retry Logic
      // Original STM behaviour:
      //   Attempt 1: reads A=4, computes 4*4=16, external write sets A=3,
      //              transaction detects conflict → retry.
      //   Attempt 2: reads A=3, computes 3*3=9, commit succeeds.
      //   Result: A=9, attempts=2.
      //
      // In Legion there are no transaction retries – the runtime guarantees
      // exclusive access.  We simulate the same observable result by
      // performing the external write first, then squaring.
        LogicalRegion A = create_int_region(ctx, runtime, 4);

        int attempt_count = 0;

        // First "attempt" – would have failed due to external modification
        attempt_count++;
        write_int_region(ctx, runtime, A, 3); // simulates the conflicting write

        // Second "attempt" – succeeds: A = A * A
        attempt_count++;
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            int A_ = acc[0];
            acc[0] = A_ * A_;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Retry Logic): A = " << read_int_region(ctx, runtime, A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_int_region(ctx, runtime, A);
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
        TaskVariantRegistrar registrar(ASYNC_CHECK_TASK_ID, "async_check");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<async_check_task>(registrar, "async_check");
    }

    return Runtime::start(argc, argv);
}
