////////////////////////////////////////////////////////////////////////////////
//  Translated from HPX/std async model to Legion task model (default mapper)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"

#include <cassert>
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
  FID_VALUE = 1
};

struct InitArgs {
  int value;
};

void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  (void)ctx;
  (void)runtime;
  assert(task->arglen == sizeof(InitArgs));
  const InitArgs *args = static_cast<const InitArgs *>(task->args);

  FieldAccessor<WRITE_DISCARD, int, 1> acc(regions[0], FID_VALUE);
  acc[Point<1>(0)] = args->value;
}

void increment_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  (void)task;
  (void)ctx;
  (void)runtime;

  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0);
  int v = acc[p];
  acc[p] = v + 1;
}

void square_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  (void)task;
  (void)ctx;
  (void)runtime;

  FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0);
  int v = acc[p];
  acc[p] = v * v;
}

int read_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime)
{
  (void)task;
  (void)ctx;
  (void)runtime;

  FieldAccessor<READ_ONLY, int, 1> acc(regions[0], FID_VALUE);
  return acc[Point<1>(0)];
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  (void)task;
  (void)regions;

  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
    return;
  }

  // Create a logical region storing one int.
  Rect<1> elem_rect(0, 0);
  IndexSpace is = runtime->create_index_space(ctx, Domain(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }
  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  auto launch_init = [&](int init_value) {
    InitArgs args{init_value};
    TaskLauncher launcher(INIT_TASK_ID, TaskArgument(&args, sizeof(args)));
    launcher.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_VALUE);
    runtime->execute_task(ctx, launcher).get_void_result();
  };

  auto launch_read = [&]() -> int {
    TaskLauncher launcher(READ_TASK_ID, TaskArgument(nullptr, 0));
    launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
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

    launch_init(0);

    Rect<1> launch_rect(0, 63);
    IndexLauncher launcher(INCREMENT_TASK_ID, Domain(launch_rect),
                           TaskArgument(nullptr, 0), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_VALUE);

    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();

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

    launch_init(2);

    Rect<1> launch_rect(0, 3);
    IndexLauncher launcher(SQUARE_TASK_ID, Domain(launch_rect),
                           TaskArgument(nullptr, 0), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_VALUE);

    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();

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
