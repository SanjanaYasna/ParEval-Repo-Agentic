// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <functional>

#include <legion.h>

using namespace Legion;

// ================================================
// Constants
// ================================================
static const double KAPPA = 3.5;
static const double LAMBDA_EXP = 4.5;
static const int NUM_STAGES = 6;

// ================================================
// Field and Task IDs
// ================================================
enum FieldIDs {
    FID_VAL = 101,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    FORCE_TASK_ID,
    AXPY_TASK_ID,
    ENERGY_TASK_ID,
};

// ================================================
// Math helpers (from system.hpp)
// ================================================
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0)
            return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double my_sign(double x) {
    return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * my_sign(x);
}

// ================================================
// Force task arguments
// ================================================
struct ForceArgs {
    size_t N;       // total system size
    bool has_left;  // whether left neighbor region is present
    bool has_right; // whether right neighbor region is present
};

// ================================================
// Force task: compute dpdt = f(q) for one block
//   Region 0: own q block (READ_ONLY)
//   Region 1: own dpdt block (WRITE_DISCARD)
//   Region 2 (if has_left): left neighbor q block (READ_ONLY)
//   Region 2 or 3 (if has_right): right neighbor q block (READ_ONLY)
// ================================================
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
    const ForceArgs &args = *(const ForceArgs *)task->args;

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    // Get boundary values from neighbor blocks
    double left_boundary = 0.0;
    double right_boundary = 0.0;
    int next_reg = 2;

    if (args.has_left) {
        const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[next_reg], FID_VAL);
        Rect<1> left_rect = runtime->get_index_space_domain(
            task->regions[next_reg].region.get_index_space());
        left_boundary = q_left_acc[left_rect.hi];
        next_reg++;
    }

    if (args.has_right) {
        const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[next_reg], FID_VAL);
        Rect<1> right_rect = runtime->get_index_space_domain(
            task->regions[next_reg].region.get_index_space());
        right_boundary = q_right_acc[right_rect.lo];
    }

    for (coord_t i = lo; i <= hi; i++) {
        double qi = q_acc[i];
        double q_left, q_right;

        if (i > lo) {
            q_left = q_acc[Point<1>(i - 1)];
        } else {
            q_left = left_boundary; // 0.0 if first block (wall), else neighbor value
        }

        if (i < hi) {
            q_right = q_acc[Point<1>(i + 1)];
        } else {
            q_right = right_boundary; // 0.0 if last block (wall), else neighbor value
        }

        dpdt_acc[i] = -signed_pow(qi, KAPPA - 1.0)
                      + signed_pow(q_left - qi, LAMBDA_EXP - 1.0)
                      - signed_pow(qi - q_right, LAMBDA_EXP - 1.0);
    }
}

// ================================================
// AXPY task: x += alpha * y  (element-wise, per block)
//   Region 0: own x block (READ_WRITE)
//   Region 1: own y block (READ_ONLY)
// ================================================
void axpy_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
    double alpha = *((const double *)task->args);

    const FieldAccessor<READ_WRITE, double, 1> x_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  y_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        x_acc[*pir] = (double)x_acc[*pir] + alpha * (double)y_acc[*pir];
    }
}

// ================================================
// Energy task: compute total Hamiltonian energy
//   Region 0: full q (READ_ONLY)
//   Region 1: full p (READ_ONLY)
// ================================================
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    size_t N = *((const size_t *)task->args);

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_VAL);

    double energy = 0.5 * checked_math::pow(std::abs((double)q_acc[0]), LAMBDA_EXP) / LAMBDA_EXP;
    for (size_t i = 0; i < N - 1; ++i) {
        double qi  = q_acc[i];
        double qi1 = q_acc[i + 1];
        double pi  = p_acc[i];
        energy += 0.5 * pi * pi
                + checked_math::pow(qi, KAPPA) / KAPPA
                + checked_math::pow(std::abs(qi - qi1), LAMBDA_EXP) / LAMBDA_EXP;
    }
    double qN = q_acc[N - 1];
    double pN = p_acc[N - 1];
    energy += 0.5 * pN * pN
            + checked_math::pow(qN, KAPPA) / KAPPA
            + 0.5 * checked_math::pow(std::abs(qN), LAMBDA_EXP) / LAMBDA_EXP;

    return energy;
}

// ================================================
// Helper: launch force as per-block tasks
// Each block task reads its own q subregion plus
// neighboring q subregions for boundary values.
// ================================================
static void launch_force(Context ctx, Runtime *runtime,
                         LogicalPartition lp_q, LogicalRegion lr_q,
                         LogicalPartition lp_dpdt, LogicalRegion lr_dpdt,
                         size_t M, size_t N) {
    for (size_t b = 0; b < M; b++) {
        ForceArgs args;
        args.N = N;
        args.has_left = (b > 0);
        args.has_right = (b < M - 1);

        TaskLauncher launcher(FORCE_TASK_ID,
                              TaskArgument(&args, sizeof(ForceArgs)));

        // Region 0: own q block (READ_ONLY)
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(lp_q, b);
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, lr_q));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 1: own dpdt block (WRITE_DISCARD)
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(lp_dpdt, b);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, lr_dpdt));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 2: left neighbor q block (READ_ONLY), if exists
        if (args.has_left) {
            LogicalRegion q_left = runtime->get_logical_subregion_by_color(lp_q, b - 1);
            launcher.add_region_requirement(
                RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, lr_q));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        // Region 2 or 3: right neighbor q block (READ_ONLY), if exists
        if (args.has_right) {
            LogicalRegion q_right = runtime->get_logical_subregion_by_color(lp_q, b + 1);
            launcher.add_region_requirement(
                RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, lr_q));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        runtime->execute_task(ctx, launcher);
    }
}

// ================================================
// Helper: launch axpy index space task  (x += alpha * y)
// ================================================
static void launch_axpy(Context ctx, Runtime *runtime,
                        const Domain &color_domain,
                        LogicalPartition lp_x, LogicalRegion lr_x,
                        LogicalPartition lp_y, LogicalRegion lr_y,
                        double alpha) {
    IndexLauncher launcher(AXPY_TASK_ID, color_domain,
                           TaskArgument(&alpha, sizeof(double)), ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(lp_x, 0, READ_WRITE, EXCLUSIVE, lr_x));
    launcher.region_requirements[0].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(lp_y, 0, READ_ONLY, EXCLUSIVE, lr_y));
    launcher.region_requirements[1].add_field(FID_VAL);
    runtime->execute_index_space(ctx, launcher);
}

// ================================================
// Top-level task
// ================================================
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // ------------------------------------------
    // Parse command-line arguments
    // ------------------------------------------
    size_t N = 1024, G = 128, steps = 100;
    double dt = 0.01;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--N") && i + 1 < command_args.argc)
            N = atoi(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--G") && i + 1 < command_args.argc)
            G = atoi(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--steps") && i + 1 < command_args.argc)
            steps = atoi(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--dt") && i + 1 < command_args.argc)
            dt = atof(command_args.argv[++i]);
    }

    const size_t M = N / G;

    // ------------------------------------------
    // Open output file
    // ------------------------------------------
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }
    outfile << "Dimension: " << N
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M
            << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    // ------------------------------------------
    // Symplectic RKN sb3a McLachlan coefficients
    // (6-stage palindromic method)
    // ------------------------------------------
    double coef_a[NUM_STAGES], coef_b[NUM_STAGES];
    coef_a[0] =  0.40518861839525227722;
    coef_a[1] = -0.28714404081652408900;
    coef_a[2] =  0.5 - coef_a[0] - coef_a[1];
    coef_a[3] =  coef_a[2];
    coef_a[4] =  coef_a[1];
    coef_a[5] =  coef_a[0];

    coef_b[0] = -3.0 / 73.0;
    coef_b[1] =  17.0 / 59.0;
    coef_b[2] =  0.5 - coef_b[0] - coef_b[1];
    coef_b[3] =  coef_b[2];
    coef_b[4] =  coef_b[1];
    coef_b[5] =  coef_b[0];

    // ------------------------------------------
    // Create index space [0, N-1] and field space
    // ------------------------------------------
    Rect<1> elem_rect(0, (coord_t)(N - 1));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(double), FID_VAL);
    }

    // ------------------------------------------
    // Create logical regions: q (positions),
    //   p (momenta), dpdt (forces)
    // ------------------------------------------
    LogicalRegion lr_q    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion lr_p    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion lr_dpdt = runtime->create_logical_region(ctx, is, fs);

    // ------------------------------------------
    // Create equal partition into M blocks of G
    // ------------------------------------------
    Rect<1> color_rect(0, (coord_t)(M - 1));
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition lp_q    = runtime->get_logical_partition(ctx, lr_q, ip);
    LogicalPartition lp_p    = runtime->get_logical_partition(ctx, lr_p, ip);
    LogicalPartition lp_dpdt = runtime->get_logical_partition(ctx, lr_dpdt, ip);

    Domain color_domain(color_rect);

    // ------------------------------------------
    // Initialize q = 0
    // ------------------------------------------
    runtime->fill_field<double>(ctx, lr_q, lr_q, FID_VAL, 0.0);

    // ------------------------------------------
    // Initialize p from random data (seed 0)
    // ------------------------------------------
    {
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));

        RegionRequirement req(lr_p, WRITE_DISCARD, EXCLUSIVE, lr_p);
        req.add_field(FID_VAL);
        InlineLauncher il(req);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        {
            const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
            for (size_t i = 0; i < N; i++)
                acc[i] = p_init[i];
        }
        runtime->unmap_region(ctx, pr);
    }

    // ------------------------------------------
    // Initialize dpdt = 0
    // ------------------------------------------
    runtime->fill_field<double>(ctx, lr_dpdt, lr_dpdt, FID_VAL, 0.0);

    // ------------------------------------------
    // Compute and print initial energy
    // ------------------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID,
                              TaskArgument(&N, sizeof(size_t)));
        launcher.add_region_requirement(
            RegionRequirement(lr_q, READ_ONLY, EXCLUSIVE, lr_q));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(lr_p, READ_ONLY, EXCLUSIVE, lr_p));
        launcher.region_requirements[1].add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // ------------------------------------------
    // Time integration loop
    // Symplectic RKN sb3a McLachlan algorithm:
    //
    // For each step, 6 stages (l = 0..5):
    //     q += a[l]*dt * p
    //     dpdt = f(q)
    //     p += b[l]*dt * dpdt
    // ------------------------------------------
    for (size_t step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {
            // q += a[l]*dt * p
            launch_axpy(ctx, runtime, color_domain,
                        lp_q, lr_q, lp_p, lr_p,
                        coef_a[l] * dt);
            // dpdt = f(q)
            launch_force(ctx, runtime,
                         lp_q, lr_q, lp_dpdt, lr_dpdt,
                         M, N);
            // p += b[l]*dt * dpdt
            launch_axpy(ctx, runtime, color_domain,
                        lp_p, lr_p, lp_dpdt, lr_dpdt,
                        coef_b[l] * dt);
        }
    }

    // ------------------------------------------
    // Compute and print final energy
    // ------------------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID,
                              TaskArgument(&N, sizeof(size_t)));
        launcher.add_region_requirement(
            RegionRequirement(lr_q, READ_ONLY, EXCLUSIVE, lr_q));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(lr_p, READ_ONLY, EXCLUSIVE, lr_p));
        launcher.region_requirements[1].add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    outfile.close();

    // ------------------------------------------
    // Clean up
    // ------------------------------------------
    runtime->destroy_logical_region(ctx, lr_q);
    runtime->destroy_logical_region(ctx, lr_p);
    runtime->destroy_logical_region(ctx, lr_dpdt);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ================================================
// Main: register tasks and start Legion runtime
// ================================================
int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(FORCE_TASK_ID, "force");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<force_task>(registrar, "force");
    }
    {
        TaskVariantRegistrar registrar(AXPY_TASK_ID, "axpy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<axpy_task>(registrar, "axpy");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
