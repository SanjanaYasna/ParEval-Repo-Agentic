// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <random>
#include <functional>
#include <cstring>
#include <cstdlib>

#include "legion.h"

using namespace Legion;

#include <boost/math/special_functions/sign.hpp>

// --- Physical constants ---
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// --- Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages, order 3) ---
// Mirrors boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
static const int NUM_STAGES = 6;
static const double coeff_a[NUM_STAGES] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - (0.40518861839525227722 + (-0.28714404081652408900)),
    0.5 - (0.40518861839525227722 + (-0.28714404081652408900)),
    -0.28714404081652408900,
    0.40518861839525227722
};
static const double coeff_b[NUM_STAGES] = {
    -3.0 / 73.0,
    17.0 / 59.0,
    1.0 - 2.0 * (17.0 / 59.0 + (-3.0 / 73.0)),
    17.0 / 59.0,
    -3.0 / 73.0,
    0.0
};

// --- IDs ---
enum FieldIDs { FID_VAL = 101 };

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_DATA_TASK_ID,
    FORCE_FIRST_TASK_ID,
    FORCE_CENTER_TASK_ID,
    FORCE_LAST_TASK_ID,
    FORCE_SINGLE_TASK_ID,
    SCALE_SUM2_TASK_ID,
    ENERGY_TASK_ID,
};

// --- Math helpers (from system.hpp) ---
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        using std::pow;
        using std::abs;
        return pow(abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

// --- Region creation helpers ---
LogicalRegion create_logical_region_1d(Context ctx, Runtime *runtime, size_t N) {
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    return runtime->create_logical_region(ctx, is, fs);
}

IndexPartition create_equal_block_partition(Context ctx, Runtime *runtime,
                                             IndexSpace is, size_t M) {
    IndexSpace color_space = runtime->create_index_space(ctx, Rect<1>(0, M - 1));
    return runtime->create_equal_partition(ctx, is, color_space);
}

inline LogicalRegion get_subregion(Context ctx, Runtime *runtime,
                                    LogicalPartition lp, size_t color) {
    return runtime->get_logical_subregion_by_color(ctx, lp,
        DomainPoint(Point<1>((coord_t)color)));
}

// --- Task: initialize to zero ---
void init_zero_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    for (PointInRectIterator<1> it(rect); it(); it++)
        acc[*it] = 0.0;
}

// --- Task: initialize from data passed as task argument ---
void init_data_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    const double *data = (const double *)task->args;
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    size_t idx = 0;
    for (PointInRectIterator<1> it(rect); it(); it++)
        acc[*it] = data[idx++];
}

// --- Force task: first block (no left neighbor) ---
// Regions: 0=q[0](READ), 1=dpdt[0](WRITE), 2=q[1](READ)
void force_first_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_r = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = q_acc[lo + (coord_t)i];

    double q_r = q_right_acc[rect_r.lo];

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[lo + (coord_t)i] = val;
    }
    dpdt_acc[lo + (coord_t)(N - 1)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// --- Force task: center block ---
// Regions: 0=q[i](READ), 1=dpdt[i](WRITE), 2=q[i-1](READ), 3=q[i+1](READ)
void force_center_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[2], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[3], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_l = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());
    Rect<1> rect_r = runtime->get_index_space_domain(ctx,
        task->regions[3].region.get_index_space());

    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = q_acc[lo + (coord_t)i];

    double q_l = q_left_acc[rect_l.hi];   // last element of left neighbor
    double q_r = q_right_acc[rect_r.lo];   // first element of right neighbor

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[lo + (coord_t)i] = val;
    }
    dpdt_acc[lo + (coord_t)(N - 1)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// --- Force task: last block (no right neighbor) ---
// Regions: 0=q[M-1](READ), 1=dpdt[M-1](WRITE), 2=q[M-2](READ)
void force_last_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_l = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = q_acc[lo + (coord_t)i];

    double q_l = q_left_acc[rect_l.hi]; // last element of left neighbor

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[lo + (coord_t)i] = val;
    }
    dpdt_acc[lo + (coord_t)(N - 1)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);
}

// --- Force task: single block (M==1, no neighbors) ---
// Regions: 0=q(READ), 1=dpdt(WRITE)
void force_single_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = q_acc[lo + (coord_t)i];

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; i++) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[lo + (coord_t)i] = val;
    }
    dpdt_acc[lo + (coord_t)(N - 1)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);
}

// --- Task: scale_sum2: target = alpha1 * target + alpha2 * source ---
// Regions: 0=target(READ_WRITE), 1=source(READ_ONLY)
// Args: double[2] = {alpha1, alpha2}
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    const double *args = (const double *)task->args;
    double alpha1 = args[0], alpha2 = args[1];

    const FieldAccessor<READ_WRITE, double, 1> target(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> source(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++)
        target[*it] = alpha1 * target[*it] + alpha2 * source[*it];
}

// --- Task: compute total energy ---
// Regions: 0=q(READ), 1=p(READ)
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N), p(N);
    for (size_t i = 0; i < N; i++) {
        q[i] = q_acc[lo + (coord_t)i];
        p[i] = p_acc[lo + (coord_t)i];
    }

    double energy = 0.5 * checked_math::pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; i++) {
        energy += 0.5 * p[i] * p[i]
                + checked_math::pow(q[i], KAPPA) / KAPPA
                + checked_math::pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    energy += 0.5 * p[N - 1] * p[N - 1]
            + checked_math::pow(q[N - 1], KAPPA) / KAPPA
            + 0.5 * checked_math::pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return energy;
}

// --- Launch helpers ---

void launch_force(Context ctx, Runtime *runtime,
                  LogicalPartition q_lp, LogicalRegion q_lr,
                  LogicalPartition dpdt_lp, LogicalRegion dpdt_lr,
                  size_t M) {
    if (M == 1) {
        TaskLauncher launcher(FORCE_SINGLE_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, 0),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, dpdt_lp, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
        return;
    }

    // First block
    {
        TaskLauncher launcher(FORCE_FIRST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, 0),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, dpdt_lp, 0),
                              WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, 1),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(FORCE_CENTER_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, i),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, dpdt_lp, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, i - 1),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, i + 1),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        TaskLauncher launcher(FORCE_LAST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, M - 1),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, dpdt_lp, M - 1),
                              WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, q_lp, M - 2),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

void launch_scale_sum2(Context ctx, Runtime *runtime,
                        LogicalPartition target_lp, LogicalRegion target_lr,
                        LogicalPartition source_lp, LogicalRegion source_lr,
                        size_t M, double alpha1, double alpha2) {
    double args[2] = {alpha1, alpha2};
    for (size_t i = 0; i < M; i++) {
        TaskLauncher launcher(SCALE_SUM2_TASK_ID,
                             TaskArgument(args, sizeof(args)));
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, target_lp, i),
                              READ_WRITE, EXCLUSIVE, target_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(get_subregion(ctx, runtime, source_lp, i),
                              READ_ONLY, EXCLUSIVE, source_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// --- Top-level task ---
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // Parse command-line arguments (--N, --G, --steps, --dt)
    size_t N = 1024, G = 128, steps = 100;
    double dt = 0.01;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--N") == 0 && i + 1 < command_args.argc)
            N = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--G") == 0 && i + 1 < command_args.argc)
            G = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--steps") == 0 && i + 1 < command_args.argc)
            steps = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--dt") == 0 && i + 1 < command_args.argc)
            dt = atof(command_args.argv[++i]);
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

    // Generate initial momentum (same RNG as original)
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // Create logical regions for q, p, dpdt (each of size N)
    LogicalRegion q_lr = create_logical_region_1d(ctx, runtime, N);
    LogicalRegion p_lr = create_logical_region_1d(ctx, runtime, N);
    LogicalRegion dpdt_lr = create_logical_region_1d(ctx, runtime, N);

    // Partition each into M blocks of size G
    IndexPartition q_ip = create_equal_block_partition(ctx, runtime,
        q_lr.get_index_space(), M);
    LogicalPartition q_lp = runtime->get_logical_partition(ctx, q_lr, q_ip);

    IndexPartition p_ip = create_equal_block_partition(ctx, runtime,
        p_lr.get_index_space(), M);
    LogicalPartition p_lp = runtime->get_logical_partition(ctx, p_lr, p_ip);

    IndexPartition dpdt_ip = create_equal_block_partition(ctx, runtime,
        dpdt_lr.get_index_space(), M);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, dpdt_ip);

    // Initialize: q = 0, p = p_init, dpdt = 0
    for (size_t i = 0; i < M; i++) {
        // q[i] = 0
        {
            TaskLauncher launcher(INIT_ZERO_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(get_subregion(ctx, runtime, q_lp, i),
                                  WRITE_DISCARD, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // p[i] from initial data
        {
            TaskLauncher launcher(INIT_DATA_TASK_ID,
                TaskArgument(&p_init[i * G], G * sizeof(double)));
            launcher.add_region_requirement(
                RegionRequirement(get_subregion(ctx, runtime, p_lp, i),
                                  WRITE_DISCARD, EXCLUSIVE, p_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // dpdt[i] = 0
        {
            TaskLauncher launcher(INIT_ZERO_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(get_subregion(ctx, runtime, dpdt_lp, i),
                                  WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }

    // Compute and report initial energy
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // Time integration: symplectic_rkn_sb3a_mclachlan
    // Algorithm per step (mirrors boost::numeric::odeint ABA ordering):
    //   q += dt*a[0]*p
    //   F(q) -> dpdt
    //   for l = 0..NUM_STAGES-2:
    //     p += dt*b[l]*dpdt;  q += dt*a[l+1]*p;  F(q)->dpdt
    //   p += dt*b[NUM_STAGES-1]*dpdt
    for (size_t step = 0; step < steps; step++) {
        // q += dt * a[0] * p
        launch_scale_sum2(ctx, runtime, q_lp, q_lr, p_lp, p_lr,
                          M, 1.0, dt * coeff_a[0]);

        // Compute force F(q) -> dpdt
        launch_force(ctx, runtime, q_lp, q_lr, dpdt_lp, dpdt_lr, M);

        for (int l = 0; l < NUM_STAGES - 1; l++) {
            // p += dt * b[l] * dpdt
            launch_scale_sum2(ctx, runtime, p_lp, p_lr, dpdt_lp, dpdt_lr,
                              M, 1.0, dt * coeff_b[l]);

            // q += dt * a[l+1] * p
            launch_scale_sum2(ctx, runtime, q_lp, q_lr, p_lp, p_lr,
                              M, 1.0, dt * coeff_a[l + 1]);

            // Compute force F(q) -> dpdt
            launch_force(ctx, runtime, q_lp, q_lr, dpdt_lp, dpdt_lr, M);
        }

        // p += dt * b[NUM_STAGES-1] * dpdt  (b[5]=0, effectively no-op)
        launch_scale_sum2(ctx, runtime, p_lp, p_lr, dpdt_lp, dpdt_lr,
                          M, 1.0, dt * coeff_b[NUM_STAGES - 1]);
    }

    // Compute and report final energy
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.region_requirements.back().add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    outfile.close();

    // Cleanup
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, q_lr);
}

// --- Main: register tasks and start Legion runtime ---
int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(INIT_ZERO_TASK_ID, "init_zero");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<init_zero_task>(registrar, "init_zero");
    }
    {
        TaskVariantRegistrar registrar(INIT_DATA_TASK_ID, "init_data");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<init_data_task>(registrar, "init_data");
    }
    {
        TaskVariantRegistrar registrar(FORCE_FIRST_TASK_ID, "force_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<force_first_task>(registrar, "force_first");
    }
    {
        TaskVariantRegistrar registrar(FORCE_CENTER_TASK_ID, "force_center");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<force_center_task>(registrar, "force_center");
    }
    {
        TaskVariantRegistrar registrar(FORCE_LAST_TASK_ID, "force_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<force_last_task>(registrar, "force_last");
    }
    {
        TaskVariantRegistrar registrar(FORCE_SINGLE_TASK_ID, "force_single");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<force_single_task>(registrar, "force_single");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
