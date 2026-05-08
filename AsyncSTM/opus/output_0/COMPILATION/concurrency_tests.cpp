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

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INCREMENT_TASK_ID,
  SQUARE_TASK_ID,
};

enum FieldIDs {
  FID_VALUE,
};

// A = A + 1
void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  int val = acc[Point<1>(0)];
  acc[Point<1>(0)] = val + 1;
}

// A = A * A
void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  int val = acc[Point<1>(0)];
  acc[Point<1>(0)] = val * val;
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
    return;
  }

  // ---------------------------------------------------------
  // TEST 1: High Concurrency Increment
  // Goal: Ensure 64 tasks can atomically increment a shared
  //       variable without data loss.
  // ---------------------------------------------------------
  {
    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    // Create a single-element logical region for the shared variable
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(int), FID_VALUE);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Initialize the region to 0 via inline mapping
    {
      RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher init_launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, init_launcher);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
      acc[Point<1>(0)] = 0;
      runtime->unmap_region(ctx, pr);
    }

    // Launch 64 increment tasks.
    // Each requires READ_WRITE on the same region, so the runtime
    // serialises them in launch order — matching STM semantics.
    for (unsigned i = 0; i < 64; ++i) {
      TaskLauncher launcher(INCREMENT_TASK_ID, TaskArgument(NULL, 0));
      launcher.add_region_requirement(
        RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
      launcher.region_requirements[0].add_field(FID_VALUE);
      runtime->execute_task(ctx, launcher);
    }

    // Read the result (implicit barrier: inline mapping waits for
    // all prior tasks with conflicting requirements to finish).
    int result;
    {
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher read_launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, read_launcher);
      pr.wait_until_valid();
      const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
      result = acc[Point<1>(0)];
      runtime->unmap_region(ctx, pr);
    }

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";

    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
  }

  // ---------------------------------------------------------
  // TEST 2: Exponential Growth (The squaring logic)
  // Goal: Test logic where the write depends heavily on the
  //       read value.
  //   2 -> 4 -> 16 -> 256 -> 65536
  // ---------------------------------------------------------
  {
    outfile << "Exponential Growth (Squaring) Test\n";
    outfile << "  Initial Value: 2\n";
    outfile << "  Threads: 4\n";
    outfile << "  Operation: A = A * A\n";

    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(int), FID_VALUE);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Initialize the region to 2
    {
      RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher init_launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, init_launcher);
      pr.wait_until_valid();
      const FieldAccessor<READ_WRITE, int, 1> acc(pr, FID_VALUE);
      acc[Point<1>(0)] = 2;
      runtime->unmap_region(ctx, pr);
    }

    // Launch 4 squaring tasks (serialised by READ_WRITE dependence)
    for (unsigned i = 0; i < 4; ++i) {
      TaskLauncher launcher(SQUARE_TASK_ID, TaskArgument(NULL, 0));
      launcher.add_region_requirement(
        RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
      launcher.region_requirements[0].add_field(FID_VALUE);
      runtime->execute_task(ctx, launcher);
    }

    // Read result
    int result;
    {
      RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(FID_VALUE);
      InlineLauncher read_launcher(req);
      PhysicalRegion pr = runtime->map_region(ctx, read_launcher);
      pr.wait_until_valid();
      const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
      result = acc[Point<1>(0)];
      runtime->unmap_region(ctx, pr);
    }

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";

    runtime->destroy_logical_region(ctx, lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
  }

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
    Runtime::preregister_task_variant<increment_task>(registrar, "increment");
  }
  {
    TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<square_task>(registrar, "square");
  }

  return Runtime::start(argc, argv);
}
