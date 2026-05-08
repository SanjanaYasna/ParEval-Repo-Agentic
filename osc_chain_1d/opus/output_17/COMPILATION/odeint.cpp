// Translation of odeint.cpp from HPX to Legion execution model
// Based on original code Copyright 2013 Mario Mulansky

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <fstream>
#include <random>
#include <functional>
#include <cstring>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ============== Constants ==============
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ============== Math helpers (from system.hpp) ==============
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

// ============== Stepper coefficients ==============
// symplectic_rkn_sb3a_mclachlan: 6 stages, order 4
// Last b coefficient is 0, so last stage is only a q update
static const int NUM_STAGES = 6;
static const double coef_a[NUM_STAGES] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - (0.40518861839525227722 - 0.28714404081652408900),
    0.5 - (0.40518861839525227722 - 0.28714404081652408900),
    -0.28714404081652408900,
    0.40518861839525227722
};
static const double coef_b[NUM_STAGES] = {
    -3.0 / 73.0,
    17.0 / 59.0,
    1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
    17.0 / 59.0,
    -3.0 / 73.0,
    0.0
};

// ============== Task and Field IDs ==============
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INIT_TASK_ID,
    FORCE_FIRST_TASK_ID,
    FORCE_CENTER_TASK_ID,
    FORCE_LAST_TASK_ID,
    UPDATE_TASK_ID,
};

enum FieldIDs {
    FID_VAL,
};

// ============== Task: Initialize a region from data in task args ==============
void init_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
    size_t len = task->arglen / sizeof(double);
    const double *data = reinterpret_cast<const double *>(task->args);

    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    size_t idx = 0;
    for (PointInRectIterator<1> it(rect); it(); it++, idx++) {
        acc[*it] = (idx < len) ? data[idx] : 0.0;
    }
}

// ============== Force task: first block ==============
// Regions: [0]=q_self(READ), [1]=q_next(READ), [2]=dpdt_self(WRITE)
void force_first_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> next_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    size_t N = rect.volume();

    // Read q into local array
    std::vector<double> q(N);
    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            q[idx] = q_acc[*it];
    }

    // Right boundary: first element of next block
    double q_r = q_next_acc[next_rect.lo];

    // Compute forces
    std::vector<double> dpdt(N);
    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);

    // Write dpdt
    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            dpdt_acc[*it] = dpdt[idx];
    }
}

// ============== Force task: center block ==============
// Regions: [0]=q_self(READ), [1]=q_prev(READ), [2]=q_next(READ), [3]=dpdt_self(WRITE)
void force_center_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_next_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> prev_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> next_rect = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    size_t N = rect.volume();

    std::vector<double> q(N);
    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            q[idx] = q_acc[*it];
    }

    double q_l = q_prev_acc[prev_rect.hi];
    double q_r = q_next_acc[next_rect.lo];

    std::vector<double> dpdt(N);
    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);

    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            dpdt_acc[*it] = dpdt[idx];
    }
}

// ============== Force task: last block ==============
// Regions: [0]=q_self(READ), [1]=q_prev(READ), [2]=dpdt_self(WRITE)
void force_last_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_prev_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> prev_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    size_t N = rect.volume();

    std::vector<double> q(N);
    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            q[idx] = q_acc[*it];
    }

    double q_l = q_prev_acc[prev_rect.hi];

    std::vector<double> dpdt(N);
    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);

    {
        size_t idx = 0;
        for (PointInRectIterator<1> it(rect); it(); it++, idx++)
            dpdt_acc[*it] = dpdt[idx];
    }
}

// ============== Update task: x += alpha * y ==============
// Regions: [0]=x(READ_WRITE), [1]=y(READ)
void update_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    assert(task->arglen == sizeof(double));
    double alpha = *reinterpret_cast<const double *>(task->args);

    const FieldAccessor<READ_WRITE, double, 1> x_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> y_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++) {
        double old_val = x_acc[*it];
        x_acc[*it] = old_val + alpha * y_acc[*it];
    }
}

// ============== Energy computation (sequential, via inline mapping) ==============
double compute_energy(const std::vector<double> &q, const std::vector<double> &p) {
    using checked_math::pow;
    const size_t N = q.size();
    double e = 0.5 * pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
            + pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

std::vector<double> read_region_data(LogicalRegion lr, Context ctx, Runtime *runtime) {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx, lr.get_index_space());

    std::vector<double> result(rect.volume());
    size_t idx = 0;
    for (PointInRectIterator<1> it(rect); it(); it++, idx++) {
        result[idx] = acc[*it];
    }

    runtime->unmap_region(ctx, pr);
    return result;
}

// ============== Helpers: launch tasks for all blocks ==============

void launch_force_tasks(size_t M, LogicalPartition q_lp, LogicalPartition dpdt_lp,
                        LogicalRegion q_lr, LogicalRegion dpdt_lr,
                        Context ctx, Runtime *runtime) {
    assert(M >= 2);

    // First block: reads q[0], q[1]; writes dpdt[0]
    {
        LogicalRegion q_sub0 = runtime->get_logical_subregion_by_color(ctx, q_lp, 0);
        LogicalRegion q_sub1 = runtime->get_logical_subregion_by_color(ctx, q_lp, 1);
        LogicalRegion dpdt_sub0 = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, 0);

        TaskLauncher launcher(FORCE_FIRST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(RegionRequirement(q_sub0, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(q_sub1, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(dpdt_sub0, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks: reads q[i-1], q[i], q[i+1]; writes dpdt[i]
    for (size_t i = 1; i < M - 1; i++) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(ctx, q_lp, i);
        LogicalRegion q_prev = runtime->get_logical_subregion_by_color(ctx, q_lp, i - 1);
        LogicalRegion q_next = runtime->get_logical_subregion_by_color(ctx, q_lp, i + 1);
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, i);

        TaskLauncher launcher(FORCE_CENTER_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(q_prev, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(q_next, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[3].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block: reads q[M-2], q[M-1]; writes dpdt[M-1]
    {
        LogicalRegion q_sub_last = runtime->get_logical_subregion_by_color(ctx, q_lp, M - 1);
        LogicalRegion q_sub_prev = runtime->get_logical_subregion_by_color(ctx, q_lp, M - 2);
        LogicalRegion dpdt_sub_last = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, M - 1);

        TaskLauncher launcher(FORCE_LAST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(RegionRequirement(q_sub_last, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(q_sub_prev, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(dpdt_sub_last, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

void launch_update_tasks(size_t M, double alpha,
                         LogicalPartition x_lp, LogicalPartition y_lp,
                         LogicalRegion x_lr, LogicalRegion y_lr,
                         Context ctx, Runtime *runtime) {
    for (size_t i = 0; i < M; i++) {
        LogicalRegion x_sub = runtime->get_logical_subregion_by_color(ctx, x_lp, i);
        LogicalRegion y_sub = runtime->get_logical_subregion_by_color(ctx, y_lp, i);

        TaskLauncher launcher(UPDATE_TASK_ID, TaskArgument(&alpha, sizeof(double)));
        launcher.add_region_requirement(RegionRequirement(x_sub, READ_WRITE, EXCLUSIVE, x_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(y_sub, READ_ONLY, EXCLUSIVE, y_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ============== Top-level task ==============
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // Parse command-line arguments
    size_t N = 1024, G = 128, steps = 100;
    double dt = 0.01;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--N") == 0 && i + 1 < command_args.argc) {
            N = atol(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--G") == 0 && i + 1 < command_args.argc) {
            G = atol(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--steps") == 0 && i + 1 < command_args.argc) {
            steps = atol(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--dt") == 0 && i + 1 < command_args.argc) {
            dt = atof(command_args.argv[++i]);
        }
    }

    const size_t M = N / G;

    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    // Create index space [0, N-1] and field space with one double field
    Rect<1> index_rect(0, (coord_t)(N - 1));
    IndexSpace is = runtime->create_index_space(ctx, index_rect);
    runtime->attach_name(is, "index_space");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    runtime->attach_name(fs, "field_space");

    // Create logical regions for q (position), p (momentum), dpdt (forces)
    LogicalRegion q_lr = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(q_lr, "q");
    runtime->attach_name(p_lr, "p");
    runtime->attach_name(dpdt_lr, "dpdt");

    // Create equal partition into M blocks of size G
    Rect<1> color_rect(0, (coord_t)(M - 1));
    IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // Generate random initial momentum (same seed/distribution as HPX version)
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // Initialize q (zeros) and p (random values) for each block
    for (size_t i = 0; i < M; i++) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(ctx, q_lp, (coord_t)i);
        LogicalRegion p_sub = runtime->get_logical_subregion_by_color(ctx, p_lp, (coord_t)i);

        // Init q block to zeros
        std::vector<double> zeros(G, 0.0);
        TaskLauncher q_init_launcher(INIT_TASK_ID,
            TaskArgument(zeros.data(), G * sizeof(double)));
        q_init_launcher.add_region_requirement(
            RegionRequirement(q_sub, WRITE_DISCARD, EXCLUSIVE, q_lr));
        q_init_launcher.region_requirements[0].add_field(FID_VAL);
        runtime->execute_task(ctx, q_init_launcher);

        // Init p block with random values
        TaskLauncher p_init_launcher(INIT_TASK_ID,
            TaskArgument(&p_init[i * G], G * sizeof(double)));
        p_init_launcher.add_region_requirement(
            RegionRequirement(p_sub, WRITE_DISCARD, EXCLUSIVE, p_lr));
        p_init_launcher.region_requirements[0].add_field(FID_VAL);
        runtime->execute_task(ctx, p_init_launcher);
    }

    // Compute and report initial energy (inline mapping waits for init tasks)
    {
        std::vector<double> q_data = read_region_data(q_lr, ctx, runtime);
        std::vector<double> p_data = read_region_data(p_lr, ctx, runtime);
        double e = compute_energy(q_data, p_data);
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // ===== Time integration: symplectic_rkn_sb3a_mclachlan =====
    // Per step algorithm (6 stages, last b coeff = 0):
    //   Stages 0..4: q += a[l]*dt*p; dpdt = f(q); p += b[l]*dt*dpdt
    //   Stage 5:     q += a[5]*dt*p  (no force/p update)
    // Legion's dependence analysis automatically enforces correct ordering.
    for (size_t step = 0; step < steps; step++) {
        // Stages 0 through 4: full q-update, force, p-update
        for (int stage = 0; stage < NUM_STAGES - 1; stage++) {
            double alpha_q = coef_a[stage] * dt;
            launch_update_tasks(M, alpha_q, q_lp, p_lp, q_lr, p_lr, ctx, runtime);

            launch_force_tasks(M, q_lp, dpdt_lp, q_lr, dpdt_lr, ctx, runtime);

            double alpha_p = coef_b[stage] * dt;
            launch_update_tasks(M, alpha_p, p_lp, dpdt_lp, p_lr, dpdt_lr, ctx, runtime);
        }

        // Stage 5: final q update only (b[5] = 0)
        {
            double alpha_q = coef_a[NUM_STAGES - 1] * dt;
            launch_update_tasks(M, alpha_q, q_lp, p_lp, q_lr, p_lr, ctx, runtime);
        }
    }

    // Compute and report final energy (inline mapping waits for all integration tasks)
    {
        std::vector<double> q_data = read_region_data(q_lr, ctx, runtime);
        std::vector<double> p_data = read_region_data(p_lr, ctx, runtime);
        double e = compute_energy(q_data, p_data);
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    outfile.close();

    // Cleanup
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ============== Main: register tasks and start Legion runtime ==============
int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(INIT_TASK_ID, "init");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<init_task>(registrar, "init");
    }
    {
        TaskVariantRegistrar registrar(FORCE_FIRST_TASK_ID, "force_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<force_first_task>(registrar, "force_first");
    }
    {
        TaskVariantRegistrar registrar(FORCE_CENTER_TASK_ID, "force_center");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<force_center_task>(registrar, "force_center");
    }
    {
        TaskVariantRegistrar registrar(FORCE_LAST_TASK_ID, "force_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<force_last_task>(registrar, "force_last");
    }
    {
        TaskVariantRegistrar registrar(UPDATE_TASK_ID, "update");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<update_task>(registrar, "update");
    }

    return Runtime::start(argc, argv);
}
