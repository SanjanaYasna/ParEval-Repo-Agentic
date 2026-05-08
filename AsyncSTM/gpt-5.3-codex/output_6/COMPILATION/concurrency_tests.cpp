////////////////////////////////////////////////////////////////////////////////
// Translated from HPX-style async test to Legion task-based execution model.
// Uses default Legion mapper (no custom mapping).
////////////////////////////////////////////////////////////////////////////////

#include <legion.h>

#include <fstream>
#include <iostream>

using namespace Legion;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1,
    INCREMENT_TASK_ID,
    SQUARE_TASK_ID
};

enum FieldIDs
{
    FID_VALUE = 1
};

static LogicalRegion create_single_int_region(Context ctx, Runtime* runtime,
                                              IndexSpace& is, FieldSpace& fs)
{
    is = runtime->create_index_space(ctx, Rect<1>(0, 0));
    fs = runtime->create_field_space(ctx);

    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_VALUE);
    }

    return runtime->create_logical_region(ctx, is, fs);
}

static void write_single_int(Context ctx, Runtime* runtime, LogicalRegion lr, int value)
{
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);

    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, int, 1> acc(pr, FID_VALUE);
    acc[Point<1>(0)] = value;

    runtime->unmap_region(ctx, pr);
}

static int read_single_int(Context ctx, Runtime* runtime, LogicalRegion lr)
{
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VALUE);

    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, int, 1> acc(pr, FID_VALUE);
    int result = acc[Point<1>(0)];

    runtime->unmap_region(ctx, pr);
    return result;
}

// A = A + 1
void increment_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
    FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    Rect<1> rect = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());

    const Point<1> p = rect.lo;
    acc[p] = acc[p] + 1;
}

// A = A * A
void square_task(const Task* task,
                 const std::vector<PhysicalRegion>& regions,
                 Context ctx, Runtime* runtime)
{
    FieldAccessor<READ_WRITE, int, 1> acc(regions[0], FID_VALUE);
    Rect<1> rect = runtime->get_index_space_domain(
        ctx, task->regions[0].region.get_index_space());

    const Point<1> p = rect.lo;
    const int v = acc[p];
    acc[p] = v * v;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime* runtime)
{
    std::ofstream outfile("concurrency_tests.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open concurrency_tests.txt" << std::endl;
        return;
    }

    // ---------------------------------------------------------
    // TEST 1: High Concurrency Increment
    // ---------------------------------------------------------
    {
        outfile << "Concurrency Increment Test\n";
        outfile << "  Initial Value: 0\n";
        outfile << "  Threads: 64\n";
        outfile << "  Operation: A = A + 1\n";

        IndexSpace is;
        FieldSpace fs;
        LogicalRegion lr = create_single_int_region(ctx, runtime, is, fs);
        write_single_int(ctx, runtime, lr, 0);

        Domain launch_domain(Rect<1>(0, 63));
        IndexLauncher launcher(INCREMENT_TASK_ID, launch_domain,
                               TaskArgument(nullptr, 0), ArgumentMap());

        launcher.add_region_requirement(
            RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        launcher.region_requirements[0].add_field(FID_VALUE);

        FutureMap fm = runtime->execute_index_space(ctx, launcher);
        fm.wait_all_results();

        int result = read_single_int(ctx, runtime, lr);

        outfile << "  Expected Result: 64\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 64 ? "PASS" : "FAIL") << "\n\n";

        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    // ---------------------------------------------------------
    // TEST 2: Exponential Growth (Squaring)
    // ---------------------------------------------------------
    {
        outfile << "Exponential Growth (Squaring) Test\n";
        outfile << "  Initial Value: 2\n";
        outfile << "  Threads: 4\n";
        outfile << "  Operation: A = A * A\n";

        IndexSpace is;
        FieldSpace fs;
        LogicalRegion lr = create_single_int_region(ctx, runtime, is, fs);
        write_single_int(ctx, runtime, lr, 2);

        Domain launch_domain(Rect<1>(0, 3));
        IndexLauncher launcher(SQUARE_TASK_ID, launch_domain,
                               TaskArgument(nullptr, 0), ArgumentMap());

        launcher.add_region_requirement(
            RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
        launcher.region_requirements[0].add_field(FID_VALUE);

        FutureMap fm = runtime->execute_index_space(ctx, launcher);
        fm.wait_all_results();

        int result = read_single_int(ctx, runtime, lr);

        outfile << "  Expected Result: 65536\n";
        outfile << "  Actual Result:   " << result << "\n";
        outfile << "  Status: " << (result == 65536 ? "PASS" : "FAIL") << "\n";

        runtime->destroy_logical_region(ctx, lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    outfile.close();
}

int main(int argc, char** argv)
{
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
