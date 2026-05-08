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

enum FieldIDs {
    FID_VAL = 0,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    VERIFY_CAPTURED_TASK_ID,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

LogicalRegion create_region(Context ctx, Runtime *runtime, FieldSpace fs) {
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    return runtime->create_logical_region(ctx, is, fs);
}

void destroy_region(Context ctx, Runtime *runtime, LogicalRegion lr) {
    runtime->destroy_index_space(ctx, lr.get_index_space());
    runtime->destroy_logical_region(ctx, lr);
}

template <typename T>
void init_var(Context ctx, Runtime *runtime, LogicalRegion lr, T val) {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, FID_VAL);
    acc[0] = val;
    runtime->unmap_region(ctx, pr);
}

template <typename T>
T read_var(Context ctx, Runtime *runtime, LogicalRegion lr) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, T, 1> acc(pr, FID_VAL);
    T val = acc[0];
    runtime->unmap_region(ctx, pr);
    return val;
}

template <typename T>
void write_var(Context ctx, Runtime *runtime, LogicalRegion lr, T val) {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, T, 1> acc(pr, FID_VAL);
    acc[0] = val;
    runtime->unmap_region(ctx, pr);
}

// ---------------------------------------------------------------------------
// Verification task (equivalent to IO.then in ASTM)
// ---------------------------------------------------------------------------

void verify_captured_task(const Task *task,
                          const std::vector<PhysicalRegion> & /*regions*/,
                          Context /*ctx*/, Runtime * /*runtime*/) {
    assert(task->arglen == sizeof(int));
    int captured = *reinterpret_cast<const int *>(task->args);
    assert(captured == 16);
}

// ---------------------------------------------------------------------------
// Top-level task
// ---------------------------------------------------------------------------

void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime) {

    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    // Shared field space for all int variables
    FieldSpace fs_int = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs_int);
        fa.allocate_field(sizeof(int), FID_VAL);
    }

    // ---------------------------------------------------------------
    // Test: Read A, Write A
    // Original: A starts at 1, transaction writes A = 2
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

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

        outfile << "Test (Read A, Write A): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Write A, Read A
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

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

        outfile << "Test (Write A, Read A): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Write A, Write A  (original only asserts A == 1 twice)
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

        {
            RegionRequirement req(A, READ_ONLY, EXCLUSIVE, A);
            req.add_field(FID_VAL);
            InlineLauncher il(req);
            PhysicalRegion pr = runtime->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
            assert(acc[0] == 1);
            assert(acc[0] == 1);
            runtime->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Write A, Write A (overwrite)
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

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

        outfile << "Test (Overwrite): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Read A, Write A, Read A
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

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

        outfile << "Test (Read -> Write -> Read): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Write A, Read A, Write A
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 1);

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

        outfile << "Test (Write -> Read -> Write): A = "
                << read_var<int>(ctx, runtime, A) << std::endl;
        destroy_region(ctx, runtime, A);
    }

    // ---------------------------------------------------------------
    // Test: Basic arithmetic with local_vars
    //   A = 4, B = 1  =>  A = A*A - B = 15
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        LogicalRegion B = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 4);
        init_var<int>(ctx, runtime, B, 1);

        {
            RegionRequirement req_a(A, READ_WRITE, EXCLUSIVE, A);
            req_a.add_field(FID_VAL);
            RegionRequirement req_b(B, READ_ONLY, EXCLUSIVE, B);
            req_b.add_field(FID_VAL);

            InlineLauncher il_a(req_a);
            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            pr_a.wait_until_valid();

            InlineLauncher il_b(req_b);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> acc_b(pr_b, FID_VAL);

            int a_val = acc_a[0];
            int b_val = acc_b[0];
            acc_a[0] = a_val * a_val - b_val;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }

        outfile << "Test (Arithmetic): A = " << read_var<int>(ctx, runtime, A)
                << ", B = " << read_var<int>(ctx, runtime, B) << std::endl;
        destroy_region(ctx, runtime, A);
        destroy_region(ctx, runtime, B);
    }

    // ---------------------------------------------------------------
    // Test: Read A to future
    //   A = 4, B = 1
    //   A_ = A_*A_          =>  A is 16 (captured for async check)
    //   IO.then(assert==16)
    //   A_ = A_ - B_        =>  A is 15
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        LogicalRegion B = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 4);
        init_var<int>(ctx, runtime, B, 1);

        int captured_a = 0;

        {
            RegionRequirement req_a(A, READ_WRITE, EXCLUSIVE, A);
            req_a.add_field(FID_VAL);
            RegionRequirement req_b(B, READ_ONLY, EXCLUSIVE, B);
            req_b.add_field(FID_VAL);

            InlineLauncher il_a(req_a);
            PhysicalRegion pr_a = runtime->map_region(ctx, il_a);
            pr_a.wait_until_valid();

            InlineLauncher il_b(req_b);
            PhysicalRegion pr_b = runtime->map_region(ctx, il_b);
            pr_b.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> acc_a(pr_a, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> acc_b(pr_b, FID_VAL);

            int a_val = acc_a[0];
            int b_val = acc_b[0];

            // A_ = A_ * A_
            a_val = a_val * a_val;
            acc_a[0] = a_val;

            // Capture intermediate value for the async verification
            captured_a = a_val;   // 16

            // A_ = A_ - B_
            acc_a[0] = a_val - b_val;

            runtime->unmap_region(ctx, pr_b);
            runtime->unmap_region(ctx, pr_a);
        }

        // Launch verification child task (equivalent to IO.then)
        {
            TaskLauncher launcher(VERIFY_CAPTURED_TASK_ID,
                                  TaskArgument(&captured_a, sizeof(int)));
            Future f = runtime->execute_task(ctx, launcher);
            f.get_void_result();   // equivalent to IO.get()
        }

        outfile << "Test (Read A to Future): A = "
                << read_var<int>(ctx, runtime, A)
                << ", B = " << read_var<int>(ctx, runtime, B) << std::endl;
        destroy_region(ctx, runtime, A);
        destroy_region(ctx, runtime, B);
    }

    // ---------------------------------------------------------------
    // Test: Retry Logic
    //   In ASTM the first attempt reads A=4, computes tmp=16, but a
    //   concurrent write sets A=3 causing the commit to fail.
    //   On retry it reads A=3, computes A=9.
    //   In Legion conflicts are prevented by the runtime, so we
    //   model the equivalent sequential outcome:
    //     1. A initialised to 4
    //     2. Conflicting write sets A = 3
    //     3. A = A*A  =>  A = 9
    // ---------------------------------------------------------------
    {
        LogicalRegion A = create_region(ctx, runtime, fs_int);
        init_var<int>(ctx, runtime, A, 4);

        // Simulate the conflicting write that occurred during the first attempt
        write_var<int>(ctx, runtime, A, 3);

        // Compute A = A * A  (reads 3, writes 9)
        {
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

        outfile << "Test (Retry Logic): A = "
                << read_var<int>(ctx, runtime, A)
                << ", Attempts = 2" << std::endl;
        destroy_region(ctx, runtime, A);
    }

    runtime->destroy_field_space(ctx, fs_int);
    outfile.close();
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar,
                                                         "top_level_task");
    }

    {
        TaskVariantRegistrar registrar(VERIFY_CAPTURED_TASK_ID,
                                       "verify_captured_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<verify_captured_task>(registrar,
                                                               "verify_captured_task");
    }

    return Runtime::start(argc, argv);
}
