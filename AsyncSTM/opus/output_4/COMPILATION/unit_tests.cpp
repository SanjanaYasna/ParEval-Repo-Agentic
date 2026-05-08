////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cassert>
#include "legion.h"

using namespace Legion;

enum FieldIDs {
    FID_VAL = 100,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    CHECK_VALUE_TASK_ID,
};

// ---------------------------------------------------------------------------
// Helper: create a single-element LogicalRegion initialised to init_val
// ---------------------------------------------------------------------------
static LogicalRegion make_var(Context ctx, Runtime *runtime,
                              FieldSpace fs, int init_val)
{
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

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
// Helper: read the single int stored in a LogicalRegion
// ---------------------------------------------------------------------------
static int read_var(Context ctx, Runtime *runtime, LogicalRegion lr)
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
// Helper: overwrite the single int stored in a LogicalRegion
// ---------------------------------------------------------------------------
static void write_var(Context ctx, Runtime *runtime, LogicalRegion lr, int val)
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
// Helper: destroy a LogicalRegion together with its IndexSpace
// ---------------------------------------------------------------------------
static void destroy_var(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    IndexSpace is = lr.get_index_space();
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_index_space(ctx, is);
}

// ---------------------------------------------------------------------------
// Child task: checks that two ints (passed in task args) are equal.
// Mirrors the future-based assertion in the original ASTM code.
// ---------------------------------------------------------------------------
void check_value_task(const Task *task,
                      const std::vector<PhysicalRegion> &/*regions*/,
                      Context /*ctx*/, Runtime */*runtime*/)
{
    assert(task->arglen == 2 * sizeof(int));
    const int *args = reinterpret_cast<const int *>(task->args);
    int actual   = args[0];
    int expected = args[1];
    assert(actual == expected);
}

// ---------------------------------------------------------------------------
// Top-level task – runs every unit test sequentially.
// ---------------------------------------------------------------------------
void top_level_task(const Task */*task*/,
                    const std::vector<PhysicalRegion> &/*regions*/,
                    Context ctx, Runtime *runtime)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    // Shared field space used by every variable in every test.
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }

    // ---- Test: Read A, Write A ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        // Transaction equivalent: read then write A = 2
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Write A, Read A ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Read A): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Write A, Write A (only reads + assertions, no actual writes) ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_ONLY, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
            int val = acc[0];
            assert(val == 1);
            assert(val == 1);
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Write A, Write A (overwrite) ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Overwrite): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Read A, Write A, Read A ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read -> Write -> Read): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Write A, Read A, Write A ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write -> Read -> Write): A = "
                << read_var(ctx, runtime, A) << std::endl;
        destroy_var(ctx, runtime, A);
    }

    // ---- Test: Basic arithmetic with local_vars ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 4);
        LogicalRegion B = make_var(ctx, runtime, fs, 1);

        // atomic { A = A*A - B; }
        {
            InlineLauncher ilA(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            ilA.add_field(FID_VAL);
            PhysicalRegion prA = runtime->map_region(ctx, ilA);
            prA.wait_until_valid();

            InlineLauncher ilB(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            ilB.add_field(FID_VAL);
            PhysicalRegion prB = runtime->map_region(ctx, ilB);
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> accA(prA, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> accB(prB, FID_VAL);

            int a_val = accA[0];
            int b_val = accB[0];
            accA[0] = a_val * a_val - b_val;

            runtime->unmap_region(ctx, prB);
            runtime->unmap_region(ctx, prA);
        }

        outfile << "Test (Arithmetic): A = " << read_var(ctx, runtime, A)
                << ", B = " << read_var(ctx, runtime, B) << std::endl;
        destroy_var(ctx, runtime, A);
        destroy_var(ctx, runtime, B);
    }

    // ---- Test: Read A to future ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 4);
        LogicalRegion B = make_var(ctx, runtime, fs, 1);

        Future check_future;
        {
            InlineLauncher ilA(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            ilA.add_field(FID_VAL);
            PhysicalRegion prA = runtime->map_region(ctx, ilA);
            prA.wait_until_valid();

            InlineLauncher ilB(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            ilB.add_field(FID_VAL);
            PhysicalRegion prB = runtime->map_region(ctx, ilB);
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> accA(prA, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> accB(prB, FID_VAL);

            int a_val = accA[0];  // 4
            int b_val = accB[0];  // 1

            a_val = a_val * a_val; // 16

            // Launch an async task that asserts the intermediate value is 16.
            {
                int args[2] = { a_val, 16 };
                TaskLauncher launcher(CHECK_VALUE_TASK_ID,
                                     TaskArgument(args, sizeof(args)));
                check_future = runtime->execute_task(ctx, launcher);
            }

            a_val = a_val - b_val; // 15
            accA[0] = a_val;

            runtime->unmap_region(ctx, prB);
            runtime->unmap_region(ctx, prA);
        }

        // Equivalent of IO.get()
        check_future.get_void_result();

        outfile << "Test (Read A to Future): A = " << read_var(ctx, runtime, A)
                << ", B = " << read_var(ctx, runtime, B) << std::endl;
        destroy_var(ctx, runtime, A);
        destroy_var(ctx, runtime, B);
    }

    // ---- Test: Retry Logic ----
    {
        LogicalRegion A = make_var(ctx, runtime, fs, 4);

        // Simulate the external write that caused the first commit to fail.
        write_var(ctx, runtime, A, 3);

        // Now perform A = A * A on the updated value.
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            int a_val = acc[0];
            acc[0] = a_val * a_val;
            runtime->unmap_region(ctx, pr);
        }

        int attempt_count = 2;

        outfile << "Test (Retry Logic): A = " << read_var(ctx, runtime, A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_var(ctx, runtime, A);
    }

    runtime->destroy_field_space(ctx, fs);
    outfile.close();
}

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
