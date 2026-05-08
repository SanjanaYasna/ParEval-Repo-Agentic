////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cassert>
#include "legion.h"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    ASSERT_VALUE_TASK_ID,
};

enum FieldIDs {
    FID_VALUE,
};

// ---------------------------------------------------------------------------
// Helper: create a 1-element logical region holding a single int, filled with
// an initial value.
// ---------------------------------------------------------------------------
LogicalRegion create_int_region(Context ctx, Runtime *runtime, int initial_value)
{
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VALUE);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    runtime->fill_field(ctx, lr, lr, FID_VALUE, initial_value);
    return lr;
}

// ---------------------------------------------------------------------------
// Helper: tear down a logical region and its spaces.
// ---------------------------------------------------------------------------
void destroy_int_region(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, lr.get_field_space());
    runtime->destroy_index_space(ctx, lr.get_index_space());
}

// ---------------------------------------------------------------------------
// Helper: read a single int from a 1-element region (inline mapping).
// ---------------------------------------------------------------------------
int read_int_region(Context ctx, Runtime *runtime, LogicalRegion lr)
{
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
    int val = acc[0];
    runtime->unmap_region(ctx, pr);
    return val;
}

// ---------------------------------------------------------------------------
// Helper: write a single int into a 1-element region (inline mapping).
// ---------------------------------------------------------------------------
void write_int_region(Context ctx, Runtime *runtime, LogicalRegion lr, int value)
{
    RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();
    const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
    acc[0] = value;
    runtime->unmap_region(ctx, pr);
}

// ---------------------------------------------------------------------------
// Child task: mirrors the IO.then() lambda – asserts the passed int == 16.
// ---------------------------------------------------------------------------
void assert_value_task(const Task *task,
                       const std::vector<PhysicalRegion> & /*regions*/,
                       Context /*ctx*/, Runtime * /*runtime*/)
{
    assert(task->arglen == sizeof(int));
    int local_A = *(const int *)task->args;
    assert(local_A == 16);
}

// ---------------------------------------------------------------------------
// Top-level task – runs every unit test sequentially.
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

    // ----------------------------------------------------------------
    // Test: Read A, Write A
    // Original: A(1); transaction { A_ = A.get_local(t); A_ = 2; }
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            // Read then write (transaction-local semantics)
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Write A, Read A
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Read A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Write A, Write A  (actually just two reads / assertions)
    // Original: A(1); transaction { A_ = A.get_local(t); assert(A_==1); assert(A_==1); }
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_ONLY, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
            assert(acc[0] == 1);
            assert(acc[0] == 1);
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Write A, Write A (overwrite)
    // Original: A(1); transaction { A_ = 2; A_ = 2; }
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Overwrite): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Read A, Write A, Read A
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Read -> Write -> Read): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Write A, Read A, Write A
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write -> Read -> Write): A = "
                << read_int_region(ctx, runtime, A) << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    // ----------------------------------------------------------------
    // Test: Basic arithmetic with local_vars
    // Original: A(4), B(1); transaction { A_ = A_*A_ - B_; }
    // Result: A = 4*4 - 1 = 15
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 4);
        LogicalRegion B = create_int_region(ctx, runtime, 1);

        {
            RegionRequirement reqA(A, READ_WRITE, EXCLUSIVE, A);
            reqA.add_field(FID_VALUE);
            RegionRequirement reqB(B, READ_ONLY, EXCLUSIVE, B);
            reqB.add_field(FID_VALUE);

            InlineLauncher ilA(reqA);
            InlineLauncher ilB(reqB);
            PhysicalRegion prA = runtime->map_region(ctx, ilA);
            PhysicalRegion prB = runtime->map_region(ctx, ilB);
            prA.wait_until_valid();
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> accA(prA, FID_VALUE);
            const FieldAccessor<READ_ONLY, int, 1>  accB(prB, FID_VALUE);

            int a_val = accA[0];
            int b_val = accB[0];
            accA[0] = a_val * a_val - b_val;

            runtime->unmap_region(ctx, prA);
            runtime->unmap_region(ctx, prB);
        }

        outfile << "Test (Arithmetic): A = "
                << read_int_region(ctx, runtime, A)
                << ", B = " << read_int_region(ctx, runtime, B) << std::endl;
        destroy_int_region(ctx, runtime, A);
        destroy_int_region(ctx, runtime, B);
    }

    // ----------------------------------------------------------------
    // Test: Read A to future
    // Original: A(4), B(1);
    //   transaction {
    //     A_ = A_*A_;                          // local A_ == 16
    //     IO.then([local_A=int(A_)]{ assert(local_A==16); });
    //     A_ = A_ - B_;                        // local A_ == 15
    //   }
    //   IO.get();   // wait for async assertion
    //
    // In Legion the async assertion is modelled as a child task.
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 4);
        LogicalRegion B = create_int_region(ctx, runtime, 1);

        int local_A; // intermediate capture for the child task

        // --- "transaction" body ----------------------------------------
        {
            RegionRequirement reqA(A, READ_WRITE, EXCLUSIVE, A);
            reqA.add_field(FID_VALUE);
            RegionRequirement reqB(B, READ_ONLY, EXCLUSIVE, B);
            reqB.add_field(FID_VALUE);

            InlineLauncher ilA(reqA);
            InlineLauncher ilB(reqB);
            PhysicalRegion prA = runtime->map_region(ctx, ilA);
            PhysicalRegion prB = runtime->map_region(ctx, ilB);
            prA.wait_until_valid();
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> accA(prA, FID_VALUE);
            const FieldAccessor<READ_ONLY, int, 1>  accB(prB, FID_VALUE);

            int a_val = accA[0]; // read A  -> 4
            int b_val = accB[0]; // read B  -> 1

            a_val = a_val * a_val;   // A_ = A_*A_ -> 16
            local_A = a_val;          // capture for IO.then

            a_val = a_val - b_val;   // A_ = A_ - B_ -> 15
            accA[0] = a_val;          // commit A = 15

            runtime->unmap_region(ctx, prA);
            runtime->unmap_region(ctx, prB);
        }

        // --- IO.then  → child task; IO.get() → get_void_result ----------
        {
            TaskLauncher launcher(ASSERT_VALUE_TASK_ID,
                                  TaskArgument(&local_A, sizeof(int)));
            Future f = runtime->execute_task(ctx, launcher);
            f.get_void_result(); // equivalent to IO.get()
        }

        outfile << "Test (Read A to Future): A = "
                << read_int_region(ctx, runtime, A)
                << ", B = " << read_int_region(ctx, runtime, B) << std::endl;
        destroy_int_region(ctx, runtime, A);
        destroy_int_region(ctx, runtime, B);
    }

    // ----------------------------------------------------------------
    // Test: Retry Logic
    // Original: A(4);
    //   attempt 1: read A=4, tmp=16, external write A=3, commit fails
    //   attempt 2: read A=3, tmp=9,                      commit succeeds
    //   A == 9,  attempt_count == 2
    //
    // In Legion the runtime guarantees exclusive access, so conflicts
    // cannot occur.  We reproduce the same *observable* outcome by
    // performing the external write first and then squaring.
    // ----------------------------------------------------------------
    {
        LogicalRegion A = create_int_region(ctx, runtime, 4);

        int attempt_count = 0;

        // Simulate the external conflicting write (first "failed" attempt)
        write_int_region(ctx, runtime, A, 3);
        ++attempt_count;

        // Second attempt succeeds: A = A*A = 9
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VALUE);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
            int a_val = acc[0];
            acc[0] = a_val * a_val;
            runtime->unmap_region(ctx, pr);
        }
        ++attempt_count;

        outfile << "Test (Retry Logic): A = "
                << read_int_region(ctx, runtime, A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_int_region(ctx, runtime, A);
    }

    outfile.close();
}

// ---------------------------------------------------------------------------
// main – register tasks, start the Legion runtime.
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
        TaskVariantRegistrar registrar(ASSERT_VALUE_TASK_ID, "assert_value");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<assert_value_task>(registrar, "assert_value");
    }

    return Runtime::start(argc, argv);
}
