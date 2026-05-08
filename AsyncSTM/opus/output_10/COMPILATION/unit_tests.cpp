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

// Equivalent of the IO.then(...) lambda that checks local_A == 16
void async_check_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(int));
  int local_A = *((const int *)task->args);
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

  // Create a shared field space for all int-valued regions
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
    fa.allocate_field(sizeof(int), FID_VAL);
  }

  // Helper: create a 1-element logical region initialized to init_val
  auto create_var = [&](int init_val) -> LogicalRegion {
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
  };

  // Helper: read an int from a region
  auto read_var = [&](LogicalRegion lr) -> int {
    InlineLauncher il(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
    int val = acc[0];
    runtime->unmap_region(ctx, pr);
    return val;
  };

  // Helper: write an int to a region
  auto write_var = [&](LogicalRegion lr, int val) {
    InlineLauncher il(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    il.add_field(FID_VAL);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();
    const FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VAL);
    acc[0] = val;
    runtime->unmap_region(ctx, pr);
  };

  // Helper: destroy a region and its index space
  auto destroy_var = [&](LogicalRegion lr) {
    runtime->destroy_index_space(ctx, lr.get_index_space());
    runtime->destroy_logical_region(ctx, lr);
  };

  // ---------------------------------------------------------------
  // Test: Read A, Write A
  // Original: shared_var<int> A(1); transaction: A_ = 2;
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

    // Transaction equivalent: read A, then write A = 2
    {
      InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
      il.add_field(FID_VAL);
      PhysicalRegion pr = runtime->map_region(ctx, il);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
      acc[0] = 2;
      runtime->unmap_region(ctx, pr);
    }

    outfile << "Test (Read A, Write A): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Write A, Read A
  // Original: shared_var<int> A(1); transaction: A_ = 2;
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

    {
      InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
      il.add_field(FID_VAL);
      PhysicalRegion pr = runtime->map_region(ctx, il);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
      acc[0] = 2;
      runtime->unmap_region(ctx, pr);
    }

    outfile << "Test (Write A, Read A): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Write A, Write A (read-only verification)
  // Original: shared_var<int> A(1); transaction: ASTM_TEST(A_ == 1) x2
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

    {
      InlineLauncher il(RegionRequirement(A, READ_ONLY, EXCLUSIVE, A));
      il.add_field(FID_VAL);
      PhysicalRegion pr = runtime->map_region(ctx, il);
      pr.wait_until_valid();
      const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VAL);
      assert(acc[0] == 1);
      assert(acc[0] == 1);
      runtime->unmap_region(ctx, pr);
    }

    outfile << "Test (Write A, Write A): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Write A, Write A (overwrite)
  // Original: shared_var<int> A(1); transaction: A_ = 2; A_ = 2;
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

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

    outfile << "Test (Overwrite): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Read A, Write A, Read A
  // Original: shared_var<int> A(1); transaction: A_ = 2;
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

    {
      InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
      il.add_field(FID_VAL);
      PhysicalRegion pr = runtime->map_region(ctx, il);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
      acc[0] = 2;
      runtime->unmap_region(ctx, pr);
    }

    outfile << "Test (Read -> Write -> Read): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Write A, Read A, Write A
  // Original: shared_var<int> A(1); transaction: A_ = 2; A_ = 2;
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(1);

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

    outfile << "Test (Write -> Read -> Write): A = " << read_var(A) << std::endl;
    destroy_var(A);
  }

  // ---------------------------------------------------------------
  // Test: Basic arithmetic with local_vars
  // Original: A(4), B(1); transaction: A_ = A_*A_ - B_;
  // Expected: A = 15, B = 1
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(4);
    LogicalRegion B = create_var(1);

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
      const FieldAccessor<READ_ONLY, int, 1>  accB(prB, FID_VAL);

      int a_val = accA[0];
      int b_val = accB[0];
      accA[0] = a_val * a_val - b_val;

      runtime->unmap_region(ctx, prB);
      runtime->unmap_region(ctx, prA);
    }

    outfile << "Test (Arithmetic): A = " << read_var(A)
            << ", B = " << read_var(B) << std::endl;
    destroy_var(A);
    destroy_var(B);
  }

  // ---------------------------------------------------------------
  // Test: Read A to future
  // Original: A(4), B(1);
  //   transaction: A_ = A_*A_;
  //                IO.then([local_A = int(A_)] { assert(local_A == 16); });
  //                A_ = A_ - B_;
  // Expected: A = 15, B = 1, async assertion passes
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(4);
    LogicalRegion B = create_var(1);

    int local_A_capture = 0;

    // Compute A = A*A, capture intermediate value, then A = A - B
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
      const FieldAccessor<READ_ONLY, int, 1>  accB(prB, FID_VAL);

      int a_val = accA[0];
      int b_val = accB[0];

      a_val = a_val * a_val;      // A_ = A_ * A_ => 16
      local_A_capture = a_val;    // captured for async check

      a_val = a_val - b_val;      // A_ = A_ - B_ => 15
      accA[0] = a_val;

      runtime->unmap_region(ctx, prB);
      runtime->unmap_region(ctx, prA);
    }

    // Launch async check task (equivalent to IO.then(...) + IO.get())
    {
      TaskLauncher launcher(ASYNC_CHECK_TASK_ID,
                            TaskArgument(&local_A_capture, sizeof(int)));
      Future f = runtime->execute_task(ctx, launcher);
      f.get_void_result();  // equivalent to IO.get()
    }

    outfile << "Test (Read A to Future): A = " << read_var(A)
            << ", B = " << read_var(B) << std::endl;
    destroy_var(A);
    destroy_var(B);
  }

  // ---------------------------------------------------------------
  // Test: Retry Logic
  // Original: A(4); first attempt reads 4, computes tmp=16, then A is
  //   externally written to 3, causing commit to fail. Second attempt
  //   reads 3, computes 3*3=9, commits.
  // Expected: A = 9, Attempts = 2
  //
  // In Legion the runtime serializes access, so we simulate the
  // equivalent sequential behaviour: write A=3 (the concurrent
  // modification), then square A.
  // ---------------------------------------------------------------
  {
    LogicalRegion A = create_var(4);

    int attempt_count = 0;

    // Simulate the first (failed) attempt's side-effect: A written to 3
    write_var(A, 3);
    attempt_count++;  // first "failed" attempt

    // Second (successful) attempt: A = A * A
    {
      InlineLauncher il(RegionRequirement(A, READ_WRITE, EXCLUSIVE, A));
      il.add_field(FID_VAL);
      PhysicalRegion pr = runtime->map_region(ctx, il);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VAL);
      int val = acc[0];
      acc[0] = val * val;
      runtime->unmap_region(ctx, pr);
    }
    attempt_count++;  // second "successful" attempt

    outfile << "Test (Retry Logic): A = " << read_var(A)
            << ", Attempts = " << attempt_count << std::endl;
    destroy_var(A);
  }

  // Cleanup the shared field space
  runtime->destroy_field_space(ctx, fs);

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
    TaskVariantRegistrar registrar(ASYNC_CHECK_TASK_ID, "async_check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<async_check_task>(registrar, "async_check");
  }

  return Runtime::start(argc, argv);
}
