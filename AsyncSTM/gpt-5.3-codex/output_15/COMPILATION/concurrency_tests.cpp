////////////////////////////////////////////////////////////////////////////////
// Translated to Legion execution model (default mapper)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <fstream>
#include <iostream>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INIT_TASK_ID,
  INCREMENT_TASK_ID,
  SQUARE_TASK_ID,
  READ_TASK_ID
};

enum FieldIDs {
  FID_VALUE = 100
};

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  const int init_value = *static_cast<const int *>(task->args);
  FieldAccessor<WRITE_DISCARD, int, 1> acc(regions[0], FID_VALUE);
  acc[Point<1>(0)] = init_value;
}

void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0);
  acc[p] = acc[p] + 1;
}

void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0);
  const int v = acc[p];
  acc[p] = v * v;
}

int read_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime)
{
  FieldAccessor<READ_ONLY, int, 1> acc(regions[0], FID_VALUE);
  return acc[Point<1>(0)];
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

  // Single-cell region holding shared integer A
  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, 0));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // ---------------------------------------------------------
  // TEST 1: High Concurrency Increment
  // ---------------------------------------------------------
  {
    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    // Init A = 0
    int initial = 0;
    TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(&initial, sizeof(initial)));
    init_launcher.add_region_requirement(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    init_launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_task(ctx, init_launcher).get_void_result();

    // Launch 64 increment tasks
    IndexLauncher inc_launcher(
        INCREMENT_TASK_ID,
        Rect<1>(0, 63),
        TaskArgument(nullptr, 0),
        ArgumentMap());
    inc_launcher.add_region_requirement(
        RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    inc_launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_index_space(ctx, inc_launcher).wait_all_results();

    // Read final result
    TaskLauncher read_launcher(READ_TASK_ID, TaskArgument(nullptr, 0));
    read_launcher.add_region_requirement(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    read_launcher.region_requirements[0].add_field(FID_VALUE);
    int result = runtime->execute_task(ctx, read_launcher).get_result<int>();

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

    // Init A = 2
    int initial = 2;
    TaskLauncher init_launcher(INIT_TASK_ID, TaskArgument(&initial, sizeof(initial)));
    init_launcher.add_region_requirement(
        RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    init_launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_task(ctx, init_launcher).get_void_result();

    // Launch 4 square tasks
    IndexLauncher sq_launcher(
        SQUARE_TASK_ID,
        Rect<1>(0, 3),
        TaskArgument(nullptr, 0),
        ArgumentMap());
    sq_launcher.add_region_requirement(
        RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    sq_launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_index_space(ctx, sq_launcher).wait_all_results();

    // Read final result
    TaskLauncher read_launcher(READ_TASK_ID, TaskArgument(nullptr, 0));
    read_launcher.add_region_requirement(
        RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    read_launcher.region_requirements[0].add_field(FID_VALUE);
    int result = runtime->execute_task(ctx, read_launcher).get_result<int>();

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";
  }

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);

  outfile.close();
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TASK_ID, "init_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_task>(registrar, "init_task");
  }
  {
    TaskVariantRegistrar registrar(INCREMENT_TASK_ID, "increment_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<increment_task>(registrar, "increment_task");
  }
  {
    TaskVariantRegistrar registrar(SQUARE_TASK_ID, "square_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<square_task>(registrar, "square_task");
  }
  {
    TaskVariantRegistrar registrar(READ_TASK_ID, "read_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int, read_task>(registrar, "read_task");
  }

  return Runtime::start(argc, argv);
}
