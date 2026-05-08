// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <functional>
#include <cassert>

#include <legion.h>

#include "system.hpp"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    FORCE_TASK_ID,
    UPDATE_TASK_ID,
};

// FID_VAL is already defined in shared_resize.hpp (included via system.hpp)

// Number of stages for symplectic_rkn_sb3a_mclachlan
static const int NUM_STAGES = 6;

// ---- Task argument structures ----

struct ForceTaskArgs {
    int block_id;
    int num_blocks;
    int block_size;
    int total_size;
};

struct UpdateTaskArgs {
    double alpha1;
    double alpha2;
};

// ---- Force computation task ----
// Reads entire q (READ_ONLY), writes own dpdt sub-region (WRITE_DISCARD)
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceTaskArgs args = *(const ForceTaskArgs *)task->args;
    int bid   = args.block_id;
    int M     = args.num_blocks;
    int G     = args.block_size;
    int N     = args.total_size;
    int base  = bid * G;

    const FieldAccessor<READ_ONLY, double, 1>    q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_VAL);

    // Left boundary coupling
    double coupling_lr;
    if (bid == 0) {
        // Fixed wall at q=0 on left
        coupling_lr = -signed_pow(q[base], LAMBDA - 1);
    } else {
        coupling_lr = -signed_pow(q[base] - q[base - 1], LAMBDA - 1);
    }

    for (int i = 0; i < G - 1; i++) {
        int gi = base + i;
        double f = -signed_pow(q[gi], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[gi] - q[gi + 1], LAMBDA - 1);
        f -= coupling_lr;
        dpdt[gi] = f;
    }

    // Last element in this block
    int last = base + G - 1;
    double f = -signed_pow(q[last], KAPPA - 1) + coupling_lr;
    if (bid == M - 1) {
        // Fixed wall at q=0 on right
        f -= signed_pow(q[last], LAMBDA - 1);
    } else {
        f -= signed_pow(q[last] - q[last + 1], LAMBDA - 1);
    }
    dpdt[last] = f;
}

// ---- Scale-sum2 update task ----
// target = alpha1 * target + alpha2 * source
void update_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    UpdateTaskArgs args = *(const UpdateTaskArgs *)task->args;
    double alpha1 = args.alpha1;
    double alpha2 = args.alpha2;

    const FieldAccessor<READ_WRITE, double, 1> target(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> source(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> itr(rect); itr(); itr++) {
        target[*itr] = alpha1 * target[*itr] + alpha2 * source[*itr];
    }
}

// ---- Energy computation (serial, via inline mapping) ----
static double compute_energy_inline(
    const FieldAccessor<READ_ONLY, double, 1> &q_acc,
    const FieldAccessor<READ_ONLY, double, 1> &p_acc,
    int N)
{
    using checked_math::pow;
    using std::abs;

    double E = 0.5 * pow(abs(q_acc[0]), LAMBDA) / LAMBDA;
    for (int i = 0; i < N - 1; ++i) {
        E += 0.5 * p_acc[i] * p_acc[i]
           + pow(q_acc[i], KAPPA) / KAPPA
           + pow(abs(q_acc[i] - q_acc[i + 1]), LAMBDA) / LAMBDA;
    }
    E += 0.5 * p_acc[N - 1] * p_acc[N - 1]
       + pow(q_acc[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q_acc[N - 1]), LAMBDA) / LAMBDA;
    return E;
}

// ---- Top-level task ----
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // --- Parse command-line arguments ---
    int N = 1024;
    int G = 128;
    int steps = 100;
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

    int M = N / G;
    assert(N % G == 0 && "N must be divisible by G");

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

    // --- Create index space [0, N-1] ---
    Rect<1> elem_rect(0, N - 1);
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
    runtime->attach_name(is, "elements_is");

    // --- Create field space ---
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    runtime->attach_name(fs, "fields_fs");

    // --- Create logical regions for q, p, dpdt ---
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(q_lr, "q_lr");
    runtime->attach_name(p_lr, "p_lr");
    runtime->attach_name(dpdt_lr, "dpdt_lr");

    // --- Create color space for M blocks ---
    Rect<1> color_rect(0, M - 1);
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);

    // --- Create equal partitions (block distribution) ---
    IndexPartition q_ip    = runtime->create_equal_partition(ctx, is, color_is);
    IndexPartition p_ip    = runtime->create_equal_partition(ctx, is, color_is);
    IndexPartition dpdt_ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, q_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, p_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, dpdt_ip);

    // --- Generate random p_init values (same seed as HPX version) ---
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // --- Initialize q = 0 ---
    {
        InlineLauncher il(RegionRequirement(q_lr, WRITE_DISCARD, EXCLUSIVE, q_lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = 0.0;
        runtime->unmap_region(ctx, pr);
    }

    // --- Initialize p from p_init ---
    {
        InlineLauncher il(RegionRequirement(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = p_init[i];
        runtime->unmap_region(ctx, pr);
    }

    // --- Initialize dpdt = 0 ---
    {
        InlineLauncher il(RegionRequirement(dpdt_lr, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = 0.0;
        runtime->unmap_region(ctx, pr);
    }

    // --- Compute initial energy ---
    {
        InlineLauncher q_il(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        q_il.add_field(FID_VAL);
        InlineLauncher p_il(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        p_il.add_field(FID_VAL);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
        q_pr.wait_until_valid();
        p_pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);
        double E = compute_energy_inline(q_acc, p_acc, N);
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    // --- Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages) ---
    // In boost odeint: coeff_a are for drift (position update q += a*dt*p)
    //                  coeff_b are for kick  (momentum update p += b*dt*dpdt)
    double coeff_a[NUM_STAGES], coeff_b[NUM_STAGES];
    coeff_a[0] =  0.40518861839525227722;
    coeff_a[1] = -0.28714404081652408900;
    coeff_a[2] =  0.5 - coeff_a[0] - coeff_a[1];
    coeff_a[3] =  coeff_a[2];
    coeff_a[4] =  coeff_a[1];
    coeff_a[5] =  coeff_a[0];

    coeff_b[0] =  0.0;
    coeff_b[1] = -3.0 / 73.0;
    coeff_b[2] =  17.0 / 59.0;
    coeff_b[3] =  1.0 - 2.0 * (coeff_b[2] + coeff_b[1]);
    coeff_b[4] =  coeff_b[2];
    coeff_b[5] =  coeff_b[1];

    // --- Helper: launch force computation for all M blocks ---
    // Each force task reads entire q (READ_ONLY) and writes its dpdt sub-region.
    // Multiple READ_ONLY tasks on the same region run in parallel.
    auto launch_force = [&]() {
        for (int i = 0; i < M; i++) {
            ForceTaskArgs fargs;
            fargs.block_id   = i;
            fargs.num_blocks = M;
            fargs.block_size = G;
            fargs.total_size = N;

            TaskLauncher launcher(FORCE_TASK_ID, TaskArgument(&fargs, sizeof(fargs)));
            // Read entire q
            launcher.add_region_requirement(
                RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            // Write dpdt sub-region i
            LogicalRegion dpdt_sub =
                runtime->get_logical_subregion_by_color(ctx, dpdt_lp, DomainPoint(Point<1>(i)));
            launcher.add_region_requirement(
                RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements[1].add_field(FID_VAL);

            runtime->execute_task(ctx, launcher);
        }
    };

    // --- Helper: launch update tasks (target = alpha1*target + alpha2*source) ---
    auto launch_update = [&](LogicalPartition target_lp, LogicalRegion target_lr_parent,
                             LogicalPartition source_lp, LogicalRegion source_lr_parent,
                             double alpha1, double alpha2) {
        for (int i = 0; i < M; i++) {
            UpdateTaskArgs uargs;
            uargs.alpha1 = alpha1;
            uargs.alpha2 = alpha2;

            TaskLauncher launcher(UPDATE_TASK_ID, TaskArgument(&uargs, sizeof(uargs)));

            LogicalRegion target_sub =
                runtime->get_logical_subregion_by_color(ctx, target_lp, DomainPoint(Point<1>(i)));
            launcher.add_region_requirement(
                RegionRequirement(target_sub, READ_WRITE, EXCLUSIVE, target_lr_parent));
            launcher.region_requirements[0].add_field(FID_VAL);

            LogicalRegion source_sub =
                runtime->get_logical_subregion_by_color(ctx, source_lp, DomainPoint(Point<1>(i)));
            launcher.add_region_requirement(
                RegionRequirement(source_sub, READ_ONLY, EXCLUSIVE, source_lr_parent));
            launcher.region_requirements[1].add_field(FID_VAL);

            runtime->execute_task(ctx, launcher);
        }
    };

    // --- Integration loop: symplectic RKN SB3A McLachlan ---
    // Boost odeint algorithm per step:
    //   for l = 0 to NUM_STAGES-1:
    //       if l > 0: q += a[l]*dt*p    (drift)
    //       dpdt = system(q)             (force)
    //       p += b[l]*dt*dpdt            (kick)
    //   q += a[0]*dt*p                   (final drift)
    //
    // Since b[0] = 0, the kick at l=0 is a no-op and the force result is unused.
    // We optimize by starting from l = 1.
    for (int step = 0; step < steps; step++) {
        for (int l = 1; l < NUM_STAGES; l++) {
            // Drift: q += a[l]*dt * p
            launch_update(q_lp, q_lr, p_lp, p_lr, 1.0, coeff_a[l] * dt);
            // Compute force: dpdt = system(q)
            launch_force();
            // Kick: p += b[l]*dt * dpdt
            launch_update(p_lp, p_lr, dpdt_lp, dpdt_lr, 1.0, coeff_b[l] * dt);
        }
        // Final drift: q += a[0]*dt * p
        launch_update(q_lp, q_lr, p_lp, p_lr, 1.0, coeff_a[0] * dt);
    }

    // --- Compute final energy ---
    {
        InlineLauncher q_il(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        q_il.add_field(FID_VAL);
        InlineLauncher p_il(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        p_il.add_field(FID_VAL);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
        q_pr.wait_until_valid();
        p_pr.wait_until_valid();
        const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);
        double E = compute_energy_inline(q_acc, p_acc, N);
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    outfile.close();

    // --- Cleanup ---
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
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
        TaskVariantRegistrar registrar(FORCE_TASK_ID, "force");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<force_task>(registrar, "force");
    }
    {
        TaskVariantRegistrar registrar(UPDATE_TASK_ID, "update");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<update_task>(registrar, "update");
    }

    return Runtime::start(argc, argv);
}
