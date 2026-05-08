// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <cstring>
#include <cstdlib>
#include <cassert>

#include "legion.h"

using namespace Legion;

// Physical constants (from system.hpp)
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Symplectic RKN SB3A McLachlan stepper coefficients (6 stages)
// Matches boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
static const int NUM_STAGES = 6;
static const double COEFF_A[NUM_STAGES] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
    -0.28714404081652408900,
    0.40518861839525227722
};
static const double COEFF_B[NUM_STAGES] = {
    -3.0 / 73.0,
    17.0 / 59.0,
    1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
    1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
    17.0 / 59.0,
    -3.0 / 73.0
};

// Task IDs
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    SYSTEM_FIRST_TASK_ID,
    SYSTEM_CENTER_TASK_ID,
    SYSTEM_LAST_TASK_ID,
    SCALE_SUM2_TASK_ID,
};

// Field IDs
enum FieldIDs {
    FID_VAL,
};

// ---------- Helper math (from system.hpp) ----------

namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double my_sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * my_sign(x);
}

// ---------- Task argument structure ----------

struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// ===== SYSTEM TASKS (from system.hpp osc_chain) =====

// First block: left boundary is fixed (q[-1]=0), right neighbor provides q_r
// regions[0] = q[0] READ_ONLY
// regions[1] = q[1] READ_ONLY (right neighbor, for first element)
// regions[2] = dpdt[0] WRITE_DISCARD
void system_first_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> bounds = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> right_bounds = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = bounds.lo[0];
    coord_t hi = bounds.hi[0];

    double q_r = q_right_acc[right_bounds.lo];

    double coupling_lr = -signed_pow(q_acc[Point<1>(lo)], LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[Point<1>(i)], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[Point<1>(i)] - q_acc[Point<1>(i + 1)], LAMBDA - 1);
        dpdt_acc[Point<1>(i)] = val - coupling_lr;
    }
    dpdt_acc[Point<1>(hi)] = -signed_pow(q_acc[Point<1>(hi)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[Point<1>(hi)] - q_r, LAMBDA - 1);
}

// Center block: has both left and right neighbors
// regions[0] = q[i] READ_ONLY
// regions[1] = q[i-1] READ_ONLY (left neighbor, for last element)
// regions[2] = q[i+1] READ_ONLY (right neighbor, for first element)
// regions[3] = dpdt[i] WRITE_DISCARD
void system_center_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_right_acc(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[3], FID_VAL);

    Rect<1> bounds = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_bounds = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> right_bounds = runtime->get_index_space_domain(ctx,
        task->regions[2].region.get_index_space());

    coord_t lo = bounds.lo[0];
    coord_t hi = bounds.hi[0];

    double q_l = q_left_acc[left_bounds.hi];
    double q_r = q_right_acc[right_bounds.lo];

    double coupling_lr = -signed_pow(q_acc[Point<1>(lo)] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[Point<1>(i)], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[Point<1>(i)] - q_acc[Point<1>(i + 1)], LAMBDA - 1);
        dpdt_acc[Point<1>(i)] = val - coupling_lr;
    }
    dpdt_acc[Point<1>(hi)] = -signed_pow(q_acc[Point<1>(hi)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[Point<1>(hi)] - q_r, LAMBDA - 1);
}

// Last block: right boundary is fixed (q[N]=0), left neighbor provides q_l
// regions[0] = q[M-1] READ_ONLY
// regions[1] = q[M-2] READ_ONLY (left neighbor, for last element)
// regions[2] = dpdt[M-1] WRITE_DISCARD
void system_last_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> q_left_acc(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[2], FID_VAL);

    Rect<1> bounds = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> left_bounds = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());

    coord_t lo = bounds.lo[0];
    coord_t hi = bounds.hi[0];

    double q_l = q_left_acc[left_bounds.hi];

    double coupling_lr = -signed_pow(q_acc[Point<1>(lo)] - q_l, LAMBDA - 1);
    for (coord_t i = lo; i < hi; i++) {
        double val = -signed_pow(q_acc[Point<1>(i)], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[Point<1>(i)] - q_acc[Point<1>(i + 1)], LAMBDA - 1);
        dpdt_acc[Point<1>(i)] = val - coupling_lr;
    }
    dpdt_acc[Point<1>(hi)] = -signed_pow(q_acc[Point<1>(hi)], KAPPA - 1)
        + coupling_lr - signed_pow(q_acc[Point<1>(hi)], LAMBDA - 1);
}

// ===== SCALE SUM2 TASK (from shared_operations.hpp) =====
// dst[i] = alpha1 * dst[i] + alpha2 * src2[i]  (in-place on dst)
// regions[0] = dst READ_WRITE
// regions[1] = src2 READ_ONLY
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    const ScaleSum2Args *args = (const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> dst_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> src2_acc(regions[1], FID_VAL);

    Rect<1> bounds = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(bounds); it(); it++) {
        dst_acc[*it] = args->alpha1 * dst_acc[*it] + args->alpha2 * src2_acc[*it];
    }
}

// ===== HELPER: command-line parsing =====

int get_int_arg(int argc, char **argv, const char *flag, int default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
    }
    return default_val;
}

double get_double_arg(int argc, char **argv, const char *flag, double default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], flag) == 0) return atof(argv[i + 1]);
    }
    return default_val;
}

// ===== HELPER: compute energy via inline mapping =====

double compute_energy(Context ctx, Runtime *runtime,
                      LogicalRegion q_lr, LogicalRegion p_lr, size_t N) {
    RegionRequirement q_req(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
    p_req.add_field(FID_VAL);

    InlineLauncher q_il(q_req);
    InlineLauncher p_il(p_req);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

    double en = 0.5 * checked_math::pow(std::abs(q_acc[Point<1>(0)]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        double qi  = q_acc[Point<1>(i)];
        double qi1 = q_acc[Point<1>(i + 1)];
        double pi  = p_acc[Point<1>(i)];
        en += 0.5 * pi * pi
            + checked_math::pow(qi, KAPPA) / KAPPA
            + checked_math::pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    double qN = q_acc[Point<1>(N - 1)];
    double pN = p_acc[Point<1>(N - 1)];
    en += 0.5 * pN * pN
        + checked_math::pow(qN, KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(qN), LAMBDA) / LAMBDA;

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return en;
}

// ===== HELPER: launch system evaluation across all M blocks =====

void launch_system(Context ctx, Runtime *runtime,
                   LogicalPartition q_lp, LogicalRegion q_lr,
                   LogicalPartition dpdt_lp, LogicalRegion dpdt_lr,
                   size_t M) {
    // First block
    {
        TaskLauncher launcher(SYSTEM_FIRST_TASK_ID, TaskArgument(NULL, 0));
        LogicalRegion q0 = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(0)));
        LogicalRegion q1 = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(1)));
        LogicalRegion d0 = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, DomainPoint(Point<1>(0)));
        launcher.add_region_requirement(RegionRequirement(q0, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(q1, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(d0, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Center blocks
    for (size_t i = 1; i < M - 1; i++) {
        TaskLauncher launcher(SYSTEM_CENTER_TASK_ID, TaskArgument(NULL, 0));
        LogicalRegion qi = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(i)));
        LogicalRegion ql = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(i - 1)));
        LogicalRegion qr = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(i + 1)));
        LogicalRegion di = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, DomainPoint(Point<1>(i)));
        launcher.add_region_requirement(RegionRequirement(qi, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(ql, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(qr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(di, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[3].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        TaskLauncher launcher(SYSTEM_LAST_TASK_ID, TaskArgument(NULL, 0));
        LogicalRegion qM = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(M - 1)));
        LogicalRegion ql = runtime->get_logical_subregion_by_color(ctx, q_lp, DomainPoint(Point<1>(M - 2)));
        LogicalRegion dM = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, DomainPoint(Point<1>(M - 1)));
        launcher.add_region_requirement(RegionRequirement(qM, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(ql, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(dM, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[2].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ===== HELPER: launch scale_sum2 across all M blocks =====
// dst = alpha1*dst + alpha2*src

void launch_scale_sum2(Context ctx, Runtime *runtime,
                       LogicalPartition dst_lp, LogicalRegion dst_lr,
                       LogicalPartition src_lp, LogicalRegion src_lr,
                       double alpha1, double alpha2, size_t M) {
    ScaleSum2Args args;
    args.alpha1 = alpha1;
    args.alpha2 = alpha2;

    for (size_t i = 0; i < M; i++) {
        TaskLauncher launcher(SCALE_SUM2_TASK_ID, TaskArgument(&args, sizeof(args)));
        LogicalRegion dst_i = runtime->get_logical_subregion_by_color(ctx, dst_lp, DomainPoint(Point<1>(i)));
        LogicalRegion src_i = runtime->get_logical_subregion_by_color(ctx, src_lp, DomainPoint(Point<1>(i)));
        launcher.add_region_requirement(RegionRequirement(dst_i, READ_WRITE, EXCLUSIVE, dst_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(RegionRequirement(src_i, READ_ONLY, EXCLUSIVE, src_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ===== TOP LEVEL TASK =====

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // Parse command-line arguments
    const InputArgs &command_args = Runtime::get_input_args();
    int argc = command_args.argc;
    char **argv = command_args.argv;

    const size_t N     = (size_t)get_int_arg(argc, argv, "--N", 1024);
    const size_t G     = (size_t)get_int_arg(argc, argv, "--G", 128);
    const size_t steps = (size_t)get_int_arg(argc, argv, "--steps", 100);
    const double dt    = get_double_arg(argc, argv, "--dt", 0.01);
    const size_t M     = N / G;

    assert(N % G == 0 && "N must be divisible by G");
    assert(M >= 2 && "Must have at least 2 blocks (N/G >= 2)");

    // Open output file
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt << std::endl;

    // Create index space [0, N-1]
    Rect<1> elem_rect(0, (coord_t)(N - 1));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);

    // Create field space with a single double field
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Create logical regions for q, p, and dpdt
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // Create equal partition into M blocks of G elements each
    Rect<1> color_rect(0, (coord_t)(M - 1));
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // Initialize q to zero and p to random values (inline mapping)
    {
        RegionRequirement q_req(q_lr, WRITE_DISCARD, EXCLUSIVE, q_lr);
        q_req.add_field(FID_VAL);
        RegionRequirement p_req(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr);
        p_req.add_field(FID_VAL);

        InlineLauncher q_il(q_req);
        InlineLauncher p_il(p_req);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
        q_pr.wait_until_valid();
        p_pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, double, 1> q_acc(q_pr, FID_VAL);
        const FieldAccessor<WRITE_DISCARD, double, 1> p_acc(p_pr, FID_VAL);

        // Same RNG setup as original HPX code
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);

        for (coord_t i = 0; i < (coord_t)N; i++) {
            q_acc[Point<1>(i)] = 0.0;
            p_acc[Point<1>(i)] = generator();
        }

        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    // Compute and report initial energy
    double init_energy = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(init_energy)) << std::endl;

    // ====== Time integration: symplectic_rkn_sb3a_mclachlan ======
    // Reproduces boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan do_step:
    //
    //   Initial A drift:
    //     q += a[0]*dt * p
    //
    //   Stage 0:
    //     sys(q, dpdt);  p += b[0]*dt * dpdt
    //
    //   Stages l = 1..5 (A drift then B kick):
    //     q += a[l]*dt * p;  sys(q, dpdt);  p += b[l]*dt * dpdt
    //

    for (size_t step = 0; step < steps; step++) {
        // Initial A drift: q += a[0]*dt*p
        launch_scale_sum2(ctx, runtime, q_lp, q_lr, p_lp, p_lr,
                          1.0, COEFF_A[0] * dt, M);

        // Stage 0: sys(q, dpdt); p += b[0]*dt*dpdt
        launch_system(ctx, runtime, q_lp, q_lr, dpdt_lp, dpdt_lr, M);
        launch_scale_sum2(ctx, runtime, p_lp, p_lr, dpdt_lp, dpdt_lr,
                          1.0, COEFF_B[0] * dt, M);

        // Stages 1 through NUM_STAGES-1
        for (int l = 1; l < NUM_STAGES; l++) {
            // A drift: q = q + a[l]*dt * p
            launch_scale_sum2(ctx, runtime, q_lp, q_lr, p_lp, p_lr,
                              1.0, COEFF_A[l] * dt, M);
            // B kick: sys(q, dpdt); p = p + b[l]*dt * dpdt
            launch_system(ctx, runtime, q_lp, q_lr, dpdt_lp, dpdt_lr, M);
            launch_scale_sum2(ctx, runtime, p_lp, p_lr, dpdt_lp, dpdt_lr,
                              1.0, COEFF_B[l] * dt, M);
        }
    }

    // Compute and report final energy
    double final_energy = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(final_energy)) << std::endl;

    outfile.close();

    // Cleanup
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ===== MAIN: register tasks and start Legion runtime =====

int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_FIRST_TASK_ID, "system_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_first_task>(registrar, "system_first");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_CENTER_TASK_ID, "system_center");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_center_task>(registrar, "system_center");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_LAST_TASK_ID, "system_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_last_task>(registrar, "system_last");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }

    return Runtime::start(argc, argv);
}
