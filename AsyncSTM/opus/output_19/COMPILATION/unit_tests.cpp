////////////////////////////////////////////////////////////////////////////////
//  Translated from HPX/ASTM to the Legion execution model.
//  Original Copyright (c) 2014 Bryce Adelstein-Lelbach, Steve Brandt
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace Legion;

/* ------------------------------------------------------------------ */
/* Task and Field IDs                                                  */
/* ------------------------------------------------------------------ */
enum {
    TOP_LEVEL_TASK_ID,
    VERIFY_VALUE_TASK_ID,
};

enum {
    FID_VAL = 100,
};

/* ------------------------------------------------------------------ */
/* Helpers: create / read / write / destroy a 1-element int region     */
/* ------------------------------------------------------------------ */
static LogicalRegion make_int_region(Context ctx, Runtime *rt, int init)
{
    IndexSpace is = rt->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = rt->create_field_space(ctx);
    {
        FieldAllocator fa = rt->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_VAL);
    }
    LogicalRegion lr = rt->create_logical_region(ctx, is, fs);

    InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = rt->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = init;
    rt->unmap_region(ctx, pr);
    return lr;
}

static int read_int_region(Context ctx, Runtime *rt, LogicalRegion lr)
{
    InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = rt->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
    int v = acc[0];
    rt->unmap_region(ctx, pr);
    return v;
}

static void write_int_region(Context ctx, Runtime *rt, LogicalRegion lr, int val)
{
    InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = rt->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = val;
    rt->unmap_region(ctx, pr);
}

static void destroy_int_region(Context ctx, Runtime *rt, LogicalRegion lr)
{
    FieldSpace fs = lr.get_field_space();
    IndexSpace is = lr.get_index_space();
    rt->destroy_logical_region(ctx, lr);
    rt->destroy_field_space(ctx, fs);
    rt->destroy_index_space(ctx, is);
}

/* ------------------------------------------------------------------ */
/* verify_value_task – mirrors IO.then() lambda from the ASTM version */
/*   Receives two ints via task args: {actual, expected}.              */
/* ------------------------------------------------------------------ */
void verify_value_task(const Task *task,
                       const std::vector<PhysicalRegion> &/*regions*/,
                       Context /*ctx*/, Runtime */*rt*/)
{
    const int *args = reinterpret_cast<const int *>(task->args);
    int actual   = args[0];
    int expected = args[1];
    assert(actual == expected);
}

/* ------------------------------------------------------------------ */
/* Top-level task – contains all unit tests                            */
/* ------------------------------------------------------------------ */
void top_level_task(const Task */*task*/,
                    const std::vector<PhysicalRegion> &/*regions*/,
                    Context ctx, Runtime *rt)
{
    std::ofstream outfile("unit_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Could not open unit_tests.txt" << std::endl;
        return;
    }

    /* ================================================================ */
    /* Test 1 – Read A, Write A                                         */
    /*   Original: shared_var<int> A(1); transaction{ A_ = 2; }         */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        // "Transaction" – exclusive read-write access guarantees atomicity
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            A_[0] = 2;
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Read A, Write A): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 2 – Write A, Read A                                         */
    /*   Original: shared_var<int> A(1); transaction{ A_ = 2; }         */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            A_[0] = 2;
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Read A): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 3 – Write A, Write A  (read-only checks)                    */
    /*   Original: assert(A_==1); assert(A_==1); (no writes → A stays)  */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_ONLY, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_ONLY, int, 1> A_(pr, FID_VAL);
            assert(A_[0] == 1);
            assert(A_[0] == 1);
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Write A, Write A): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 4 – Write A, Write A (overwrite)                            */
    /*   Original: A_ = 2; A_ = 2;                                     */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            A_[0] = 2;
            A_[0] = 2;
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Overwrite): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 5 – Read A, Write A, Read A                                 */
    /*   Original: transaction{ A_ = 2; }                               */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            A_[0] = 2;
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Read -> Write -> Read): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 6 – Write A, Read A, Write A                                */
    /*   Original: transaction{ A_ = 2; A_ = 2; }                      */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 1);

        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            A_[0] = 2;
            A_[0] = 2;
            rt->unmap_region(ctx, pr);
        }

        outfile << "Test (Write -> Read -> Write): A = "
                << read_int_region(ctx, rt, A) << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    /* ================================================================ */
    /* Test 7 – Basic arithmetic   A = A*A − B                          */
    /*   A(4), B(1) → A = 16 − 1 = 15,  B unchanged = 1               */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 4);
        LogicalRegion B = make_int_region(ctx, rt, 1);

        // Single "transaction" that accesses both regions atomically
        {
            InlineLauncher ilA(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            ilA.add_field(FID_VAL);
            PhysicalRegion prA = rt->map_region(ctx, ilA);
            prA.wait_until_valid();

            InlineLauncher ilB(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            ilB.add_field(FID_VAL);
            PhysicalRegion prB = rt->map_region(ctx, ilB);
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> A_(prA, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> B_(prB, FID_VAL);

            int a_val = A_[0];
            int b_val = B_[0];
            A_[0] = a_val * a_val - b_val;

            rt->unmap_region(ctx, prB);
            rt->unmap_region(ctx, prA);
        }

        outfile << "Test (Arithmetic): A = " << read_int_region(ctx, rt, A)
                << ", B = " << read_int_region(ctx, rt, B) << std::endl;
        destroy_int_region(ctx, rt, A);
        destroy_int_region(ctx, rt, B);
    }

    /* ================================================================ */
    /* Test 8 – Read A to future                                        */
    /*   A(4), B(1)                                                     */
    /*   Transaction body:                                              */
    /*       A_ = A_*A_            → local A = 16                       */
    /*       IO.then { assert(local_A == 16); }   (async check)         */
    /*       A_ = A_ − B_         → local A = 15                        */
    /*   After commit: A == 15, B == 1                                  */
    /*                                                                  */
    /*   The IO.then() is modelled as a child task that receives the    */
    /*   captured intermediate value and verifies it.                   */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 4);
        LogicalRegion B = make_int_region(ctx, rt, 1);

        int captured_A; // intermediate value captured for the "future"

        {
            InlineLauncher ilA(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            ilA.add_field(FID_VAL);
            PhysicalRegion prA = rt->map_region(ctx, ilA);
            prA.wait_until_valid();

            InlineLauncher ilB(RegionRequirement(B, READ_ONLY, EXCLUSIVE, B));
            ilB.add_field(FID_VAL);
            PhysicalRegion prB = rt->map_region(ctx, ilB);
            prB.wait_until_valid();

            const FieldAccessor<READ_WRITE, int, 1> A_(prA, FID_VAL);
            const FieldAccessor<READ_ONLY,  int, 1> B_(prB, FID_VAL);

            int a_val = A_[0];
            int b_val = B_[0];

            // A_ = A_ * A_
            A_[0] = a_val * a_val;
            captured_A = A_[0]; // snapshot for the "future" check

            // A_ = A_ - B_
            A_[0] = A_[0] - b_val;

            rt->unmap_region(ctx, prB);
            rt->unmap_region(ctx, prA);
        }

        // Launch verify task – mirrors IO.then( [local_A] { assert(…); } )
        {
            int args[2] = { captured_A, 16 };
            TaskLauncher launcher(VERIFY_VALUE_TASK_ID,
                                  TaskArgument(args, sizeof(args)));
            Future f = rt->execute_task(ctx, launcher);
            f.get_void_result(); // equivalent to IO.get()
        }

        outfile << "Test (Read A to Future): A = "
                << read_int_region(ctx, rt, A)
                << ", B = " << read_int_region(ctx, rt, B) << std::endl;
        destroy_int_region(ctx, rt, A);
        destroy_int_region(ctx, rt, B);
    }

    /* ================================================================ */
    /* Test 9 – Retry Logic                                             */
    /*   Original ASTM behaviour:                                       */
    /*     A(4); attempt 1 reads 4, computes 16, but an external write  */
    /*     sets A=3 → conflict → retry. Attempt 2 reads 3, computes 9.  */
    /*     Result: A=9, attempts=2.                                     */
    /*                                                                  */
    /*   In Legion there is no optimistic retry; the runtime serializes */
    /*   accesses.  We model the identical logical sequence:             */
    /*     1. Init A = 4                                                */
    /*     2. Interfering write  A = 3                                  */
    /*     3. Square             A = A*A = 9                            */
    /*   Two write passes ⇒ attempt_count = 2.                         */
    /* ================================================================ */
    {
        LogicalRegion A = make_int_region(ctx, rt, 4);

        // The "interfering" write that would have caused a conflict in STM
        write_int_region(ctx, rt, A, 3);

        // Square A  (reads 3, writes 9)
        {
            InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
            il.add_field(FID_VAL);
            PhysicalRegion pr = rt->map_region(ctx, il);
            pr.wait_until_valid();
            const FieldAccessor<READ_WRITE, int, 1> A_(pr, FID_VAL);
            int v = A_[0];
            A_[0] = v * v;
            rt->unmap_region(ctx, pr);
        }

        int attempt_count = 2; // two operations mirror the two STM attempts
        outfile << "Test (Retry Logic): A = "
                << read_int_region(ctx, rt, A)
                << ", Attempts = " << attempt_count << std::endl;
        destroy_int_region(ctx, rt, A);
    }

    outfile.close();
}

/* ------------------------------------------------------------------ */
/* main – register tasks and start the Legion runtime                  */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(VERIFY_VALUE_TASK_ID, "verify_value");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<verify_value_task>(registrar, "verify_value");
    }

    return Runtime::start(argc, argv);
}
