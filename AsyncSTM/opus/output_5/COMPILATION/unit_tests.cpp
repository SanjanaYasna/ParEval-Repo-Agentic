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

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    FUTURE_CHECK_TASK_ID,
};

enum FieldIDs {
    FID_VAL,
};

// ---------------------------------------------------------------------------
// Helper: create a single-element LogicalRegion with one int field
// ---------------------------------------------------------------------------
static LogicalRegion create_int_region(Context ctx, Runtime *runtime)
{
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    return runtime->create_logical_region(ctx, is, fs);
}

// ---------------------------------------------------------------------------
// Helper: write an int into a single-element region (WRITE_DISCARD)
// ---------------------------------------------------------------------------
static void write_region(Context ctx, Runtime *runtime,
                         LogicalRegion lr, int val)
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
// Helper: read an int from a single-element region (READ_ONLY)
// ---------------------------------------------------------------------------
static int read_region(Context ctx, Runtime *runtime, LogicalRegion lr)
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
// Helper: destroy a region and its index/field spaces
// ---------------------------------------------------------------------------
static void destroy_region(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, lr.get_field_space());
    runtime->destroy_index_space(ctx, lr.get_index_space());
}

// ---------------------------------------------------------------------------
// Small task used by the "Read A to future" test – mirrors the original
// IO.then( [local_A](...){ ASTM_TEST(local_A == 16); } )
// ---------------------------------------------------------------------------
void future_check_task(const Task *task,
                       const std::vector<PhysicalRegion> & /*regions*/,
                       Context /*ctx*/, Runtime * /*runtime*/)
{
    assert(task->arglen == sizeof(int));
    int local_A = *reinterpret_cast<const int *>(task->args);
    assert(local_A == 16);
}

// ---------------------------------------------------------------------------
// Top-level task – contains all the unit tests
// ---------------------------------------------------------------------------
void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    // ===================================================================
    // Test 1 – Read A, Write A
    // Original: shared_var<int> A(1); transaction { A_ = 2; }
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

        // "transaction" – exclusive inline map, read then write
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            // read (implicit via get_local) then write
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 2 – Write A, Read A
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

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
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 3 – Write A, Write A  (only reads, both assert ==1)
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

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
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 4 – Write A, Write A (overwrite)
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

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
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 5 – Read A, Write A, Read A
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

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
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 6 – Write A, Read A, Write A
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 1);

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
                << read_region(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ===================================================================
    // Test 7 – Basic arithmetic with local_vars   A = A*A - B
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        LogicalRegion B = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 4);
        write_region(ctx, runtime, B, 1);

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
            accA[0] = a_val * a_val - b_val;   // 4*4 - 1 = 15

            runtime->unmap_region(ctx, prB);
            runtime->unmap_region(ctx, prA);
        }

        outfile << "Test (Arithmetic): A = "
                << read_region(ctx, runtime, A)
                << ", B = "
                << read_region(ctx, runtime, B) << std::endl;
        destroy_region(ctx, runtime, A);
        destroy_region(ctx, runtime, B);
    }

    // ===================================================================
    // Test 8 – Read A to future
    //   Original: A=A*A, capture local_A (==16), async assert ==16,
    //             then A = A - B  →  final A = 15
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        LogicalRegion B = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 4);
        write_region(ctx, runtime, B, 1);

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

            int a_val = accA[0];       // 4
            int b_val = accB[0];       // 1

            a_val = a_val * a_val;     // 16
            accA[0] = a_val;

            // Capture snapshot for the async check
            int local_A = a_val;       // 16

            // Launch a child task that asserts local_A == 16
            TaskLauncher launcher(FUTURE_CHECK_TASK_ID,
                                  TaskArgument(&local_A, sizeof(int)));
            Future f = runtime->execute_task(ctx, launcher);

            // Continue the "transaction": A = A - B
            accA[0] = a_val - b_val;   // 15

            runtime->unmap_region(ctx, prB);
            runtime->unmap_region(ctx, prA);

            // Wait for the future-based check (mirrors IO.get())
            f.get_void_result();
        }

        outfile << "Test (Read A to Future): A = "
                << read_region(ctx, runtime, A)
                << ", B = "
                << read_region(ctx, runtime, B) << std::endl;
        destroy_region(ctx, runtime, A);
        destroy_region(ctx, runtime, B);
    }

    // ===================================================================
    // Test 9 – Retry logic
    //   Original: A(4), first attempt externally writes A=3 causing the
    //   STM commit to fail.  Second attempt reads 3, computes 3*3=9.
    //   In Legion there is no optimistic retry, so we simulate the
    //   same control flow: on the first pass we write 3 and skip the
    //   computed value; on the second pass we commit normally.
    // ===================================================================
    {
        LogicalRegion A = create_int_region(ctx, runtime);
        write_region(ctx, runtime, A, 4);

        bool fail      = true;
        int  attempt_count = 0;
        bool committed = false;

        while (!committed) {
            ++attempt_count;

            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);

            int a_val = acc[0];
            int tmp   = a_val * a_val;

            if (fail) {
                // Simulate external modification that causes a conflict
                acc[0] = 3;
                fail = false;
                runtime->unmap_region(ctx, pr);
                continue;   // transaction fails – retry
            }

            acc[0] = tmp;               // second pass: 3*3 = 9
            runtime->unmap_region(ctx, pr);
            committed = true;
        }

        outfile << "Test (Retry Logic): A = "
                << read_region(ctx, runtime, A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_region(ctx, runtime, A);
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
        TaskVariantRegistrar registrar(FUTURE_CHECK_TASK_ID, "future_check");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<future_check_task>(registrar, "future_check");
    }

    return Runtime::start(argc, argv);
}
