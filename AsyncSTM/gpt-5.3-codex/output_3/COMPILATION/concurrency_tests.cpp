#include "legion.h"

#include <fstream>
#include <iostream>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INCREMENT_TASK_ID,
  SQUARE_TASK_ID
};

enum FieldIDs {
  FID_VALUE = 100
};

struct SharedIntRegion {
  IndexSpace is;
  FieldSpace fs;
  LogicalRegion lr;
};

static SharedIntRegion create_shared_int_region(Context ctx, Runtime* runtime, int initial_value) {
  SharedIntRegion r;

  r.is = runtime->create_index_space(ctx, Rect<1>(0, 0));
  r.fs = runtime->create_field_space(ctx);

  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, r.fs);
    allocator.allocate_field(sizeof(int), FID_VALUE);
  }

  r.lr = runtime->create_logical_region(ctx, r.is, r.fs);

  runtime->fill_field(ctx, r.lr, r.lr, FID_VALUE, &initial_value, sizeof(initial_value));
  return r;
}

static void destroy_shared_int_region(Context ctx, Runtime* runtime, const SharedIntRegion& r) {
  runtime->destroy_logical_region(ctx, r.lr);
  runtime->destroy_field_space(ctx, r.fs);
  runtime->destroy_index_space(ctx, r.is);
}

static int read_shared_int(Context ctx, Runtime* runtime, LogicalRegion lr) {
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(FID_VALUE);

  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  const FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
  int value = acc[Point<1>(0)];

  runtime->unmap_region(ctx, pr);
  return value;
}

// A = A + 1
void increment_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime) {
  (void)task;
  (void)ctx;
  (void)runtime;
  const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0); // single element
  acc[p] = acc[p] + 1;
}

// A = A * A
void square_task(const Task* task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx, Runtime* runtime) {
  (void)task;
  (void)ctx;
  (void)runtime;
  const FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
  const Point<1> p(0); // single element
  int v = acc[p];
  acc[p] = v * v;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime* runtime) {
  std::ofstream outfile("concurrency_tests.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
    return;
  }

  // ---------------------------------------------------------
  // TEST 1: High Concurrency Increment
  // Goal: Ensure 64 tasks can atomically increment shared state.
  // ---------------------------------------------------------
  {
    outfile << "Concurrency Increment Test\n";
    outfile << "  Initial Value: 0\n";
    outfile << "  Threads: 64\n";
    outfile << "  Operation: A = A + 1\n";

    SharedIntRegion data = create_shared_int_region(ctx, runtime, 0);

    IndexSpace launch_is = runtime->create_index_space(ctx, Rect<1>(0, 63));
    ArgumentMap arg_map;
    IndexLauncher launcher(INCREMENT_TASK_ID, launch_is, TaskArgument(nullptr, 0), arg_map);

    launcher.add_region_requirement(RegionRequirement(data.lr, READ_WRITE, EXCLUSIVE, data.lr));
    launcher.region_requirements[0].add_field(FID_VALUE);

    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();

    int result = read_shared_int(ctx, runtime, data.lr);

    outfile << "  Expected Result: 64\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";

    runtime->destroy_index_space(ctx, launch_is);
    destroy_shared_int_region(ctx, runtime, data);
  }

  // ---------------------------------------------------------
  // TEST 2: Exponential Growth (Squaring)
  // Goal: Run 4 squaring transactions.
  // 2 -> 4 -> 16 -> 256 -> 65536
  // ---------------------------------------------------------
  {
    outfile << "Exponential Growth (Squaring) Test\n";
    outfile << "  Initial Value: 2\n";
    outfile << "  Threads: 4\n";
    outfile << "  Operation: A = A * A\n";

    SharedIntRegion data = create_shared_int_region(ctx, runtime, 2);

    IndexSpace launch_is = runtime->create_index_space(ctx, Rect<1>(0, 3));
    ArgumentMap arg_map;
    IndexLauncher launcher(SQUARE_TASK_ID, launch_is, TaskArgument(nullptr, 0), arg_map);

    launcher.add_region_requirement(RegionRequirement(data.lr, READ_WRITE, EXCLUSIVE, data.lr));
    launcher.region_requirements[0].add_field(FID_VALUE);

    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();

    int result = read_shared_int(ctx, runtime, data.lr);

    outfile << "  Expected Result: 65536\n";
    outfile << "  Actual Result:   " << result << "\n";
    outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";

    runtime->destroy_index_space(ctx, launch_is);
    destroy_shared_int_region(ctx, runtime, data);
  }

  outfile.close();
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
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

  return Runtime::start(argc, argv);
}
