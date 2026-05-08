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

// Task equivalent of the IO.then lambda that checks intermediate_A == 16
void future_check_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    assert(task->arglen == sizeof(int));
    int local_A = *(reinterpret_cast<const int *>(task->args));
    assert(local_A == 16);
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    // Shared field space for all int-valued variables
    FieldSpace int_fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, int_fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }

    // Helper: create a 1-element logical region initialized with init_val
    // (equivalent to shared_var<int> A(init_val))
    auto create_int_var = [&](int init_val) -> LogicalRegion {
        IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
        LogicalRegion lr = runtime->create_logical_region(ctx, is, int_fs);

        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
        acc[0] = init_val;
        runtime->unmap_region(ctx, pr);
        return lr;
    };

    // Helper: read an int from a 1-element region (equivalent to A.read())
    auto read_int = [&](LogicalRegion lr) -> int {
        RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
        int val = acc[0];
        runtime->unmap_region(ctx, pr);
        return val;
    };

    // Helper: write an int to a 1-element region (equivalent to A.write(val))
    auto write_int = [&](LogicalRegion lr, int val) {
        RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
        acc[0] = val;
        runtime->unmap_region(ctx, pr);
    };

    // Helper: destroy a logical region and its index space
    auto destroy_var = [&](LogicalRegion lr) {
        IndexSpace is = lr.get_index_space();
        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_index_space(ctx, is);
    };

    // ---- Test: Read A, Write A ----
    // Original: shared_var<int> A(1); transaction { A_ = 2; }
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Read A, Write A): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Write A, Read A ----
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Write A, Read A): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Write A, Write A (assert-only reads) ----
    // Original: ASTM_TEST(A_ == 1); ASTM_TEST(A_ == 1);
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_ONLY, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
            int val = acc[0];
            assert(val == 1);
            assert(val == 1);
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Write A, Write A): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Write A, Write A (overwrite) ----
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Overwrite): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Read A, Write A, Read A ----
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Read -> Write -> Read): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Write A, Read A, Write A ----
    {
        LogicalRegion A = create_int_var(1);
        {
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            acc[0] = 2;
            acc[0] = 2;
            runtime->unmap_region(ctx, pr);
        }
        outfile << "Test (Write -> Read -> Write): A = " << read_int(A) << std::endl;
        destroy_var(A);
    }

    // ---- Test: Basic arithmetic with local_vars ----
    // Original: atomic { A = A*A - B; }  with A=4, B=1 => A=15
    {
        LogicalRegion A = create_int_var(4);
        LogicalRegion B = create_int_var(1);
        {
            RegionRequirement req_a(A, READ_WRITE, EXCLUSIVE, A);
            req_a.add_field(FID_VAL);
            InlineLauncher il_a(req_a);
            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            pr_a.wait_until_valid();

            RegionRequirement req_b(B, READ_ONLY, EXCLUSIVE, B);
            req_b.add_field(FID_VAL);
            InlineLauncher il_b(req_b);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY, int, 1>  acc_b(pr_b, FID_VAL);

            int a_val = acc_a[0];
            int b_val = acc_b[0];
            acc_a[0] = a_val * a_val - b_val;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }
        outfile << "Test (Arithmetic): A = " << read_int(A)
                << ", B = " << read_int(B) << std::endl;
        destroy_var(A);
        destroy_var(B);
    }

    // ---- Test: Read A to future ----
    // Original: A=4, B=1; A=A*A; IO.then(check A==16); A=A-B; => A=15
    {
        LogicalRegion A = create_int_var(4);
        LogicalRegion B = create_int_var(1);

        int intermediate_A;
        {
            RegionRequirement req_a(A, READ_WRITE, EXCLUSIVE, A);
            req_a.add_field(FID_VAL);
            InlineLauncher il_a(req_a);
            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            pr_a.wait_until_valid();

            RegionRequirement req_b(B, READ_ONLY, EXCLUSIVE, B);
            req_b.add_field(FID_VAL);
            InlineLauncher il_b(req_b);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY, int, 1>  acc_b(pr_b, FID_VAL);

            int a_val = acc_a[0];
            int b_val = acc_b[0];

            // A = A * A
            a_val = a_val * a_val;   // 16
            intermediate_A = a_val;  // capture for future check

            // A = A - B
            a_val = a_val - b_val;   // 15
            acc_a[0] = a_val;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }

        // Launch async check task (equivalent to IO.then / IO.get)
        {
            TaskLauncher launcher(FUTURE_CHECK_TASK_ID,
                                  TaskArgument(&intermediate_A, sizeof(int)));
            Future f = runtime->execute_task(ctx, launcher);
            f.get_void_result();  // equivalent to IO.get()
        }

        outfile << "Test (Read A to Future): A = " << read_int(A)
                << ", B = " << read_int(B) << std::endl;
        destroy_var(A);
        destroy_var(B);
    }

    // ---- Test: Retry Logic ----
    // In ASTM, a direct write to A during a transaction causes read-validation
    // failure and retry.  In Legion the runtime manages data ordering, so no
    // optimistic retry is needed.  We model the interference as a sequential
    // write (A = 3) followed by the squaring computation, yielding A = 9.
    {
        LogicalRegion A = create_int_var(4);

        // Interference write (equivalent to A.write(3) during first ASTM attempt)
        write_int(A, 3);

        // Compute A = A * A  (reads 3, writes 9)
        int attempt_count = 0;
        {
            ++attempt_count;
            RegionRequirement req(A, READ_WRITE, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
            int val = acc[0];
            acc[0] = val * val;
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Retry Logic): A = " << read_int(A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_var(A);
    }

    runtime->destroy_field_space(ctx, int_fs);
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
        TaskVariantRegistrar registrar(FUTURE_CHECK_TASK_ID, "future_check");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<future_check_task>(registrar, "future_check");
    }

    return Runtime::start(argc, argv);
}
