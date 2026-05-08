////////////////////////////////////////////////////////////////////////////////
//  Translated to Legion execution model (default mapper)
//  Original HPX/ASTM concurrency tests replaced with Legion tasks + logical regions
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  SET_VALUE_TASK_ID,
  INCREMENT_TASK_ID,
  SQUARE_TASK_ID,
  READ_VALUE_TASK_ID
};

enum FieldIDs {
  FID_VALUE = 1
};

void set_value_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  assert(task->arglen == sizeof(int));
  const int value = *static_cast<const int*>(task->args);

  FieldAccessor<WRITE_DISCARD, int, 1> acc(regions[0], FID_VALUE);
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  // Single-element region, but keep generic loop.
  for (PointInRectIterator<1> p(rect); p(); p++) {
    acc[*p] = value;
  }
}

void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  for (PointInRectIterator<1> p(rect); p(); p++) {
    acc[*p] = acc[*p] + 1;
  }
}

void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  for (PointInRectIterator<1> p(rect); p(); p++) {
    const int v = acc[*p];
    acc[*p] = v * v;
  }
}

int read_value_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_ONLY, int, 1> acc(regions[0], FID_VALUE);
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  // Single element expected.
  return acc[rect.lo];
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime *runtime)
{
  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
    return;
  }

  // Shared integer stored in a 1-element logical region.
  Rect<1> elem_rect(Point<1>(0), Point<1>(0));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  auto launch_set = [&](int value) {
    TaskLauncher launcher(SET_VALUE_TASK_ID, TaskArgument(&value, sizeof(value)));
    launcher.add_region_requirement(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_task(ctx, launcher).get_void_result();
  };

  auto launch_many = [&](TaskID tid, int count) {
    std::vector<Future> futures;
    futures.reserve(count);

    for (int i = 0; i < count; ++i) {
      TaskLauncher launcher(tid, TaskArgument(nullptr, 0));
      launcher.add_region_requirement(
          RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
      launcher.region_requirements[0].add_field(FID_VALUE);
      futures.push_back(runtime->execute_task(ctx, launcher));
    }

    for (auto &f : futures) {
      f.get_void_result();
    }
  };

  auto launch_read = [&]() -> int {
    TaskLauncher launcher(READ_VALUE_TASK_ID, TaskArgument(nullptr, 0));
    launcher.add_region_requirement(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_VALUE);
    return runtime->execute_task(ctx, launcher).get_result<int>();
  };

  // ---------------------------------------------------------
  // TEST 1: High Concurrency Increment
  // ---------------------------------------------------------
  {
    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    launch_set(0);
    launch_many(INCREMENT_TASK_ID, 64);
    int result = launch_read();

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";
  }

  // ---------------------------------------------------------
  // TEST 2: Exponential Growth (Squaring)
  // ---------------------------------------------------------
  {
    outfile << "Exponential Growth (Squaring) Test\n";
    outfile << "  Initial Value: 2\n";
    outfile << "  Threads: 4\n";
    outfile << "  Operation: A = A * A\n";

    launch_set(2);
    launch_many(SQUARE_TASK_ID, 4);
    int result = launch_read();

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
  }

  outfile.close();

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
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
    TaskVariantRegistrar registrar(SET_VALUE_TASK_ID, "set_value");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<set_value_task>(registrar, "set_value");
  }
  {
    TaskVariantRegistrar registrar(INCREMENT_TASK_ID, "increment");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<increment_task>(registrar, "increment");
  }
  {
    TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<square_task>(registrar, "square");
  }
  {
    TaskVariantRegistrar registrar(READ_VALUE_TASK_ID, "read_value");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, read_value_task>(registrar, "read_value");
  }

  return Runtime::start(argc, argv);
}
