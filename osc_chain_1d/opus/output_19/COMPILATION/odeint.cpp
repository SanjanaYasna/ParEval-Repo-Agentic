// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <legion.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <random>
#include <functional>
#include <fstream>
#include <cassert>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ============================================================
// Task IDs
// ============================================================
enum {
    TOP_LEVEL_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_DATA_TASK_ID,
    SCALE_SUM2_TASK_ID,
    SYSTEM_FIRST_TASK_ID,
    SYSTEM_CENTER_TASK_ID,
    SYSTEM_LAST_TASK_ID,
    EXTRACT_FIRST_TASK_ID,
    EXTRACT_LAST_TASK_ID,
    ENERGY_TASK_ID,
};

enum {
    FID_VAL = 0,
};

// ============================================================
// Physical constants (from system.hpp)
// ============================================================
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ============================================================
// Math helpers (from system.hpp)
// ============================================================
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

// ============================================================
// Symplectic RKN SB3A McLachlan stepper coefficients (6 stages)
// From boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
// ============================================================
static const int NUM_STAGES = 6;

static const double COEF_A[NUM_STAGES] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
    -0.28714404081652408900,
    0.40518861839525227722
};

static const double COEF_B[NUM_STAGES] = {
    -3.0 / 73.0,
    17.0 / 59.0,
    1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
    17.0 / 59.0,
    -3.0 / 73.0,
    0.0
};

// ============================================================
// Task argument structures
// ============================================================
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// ============================================================
// Task: Initialize a region to zero
// Region 0: target (WRITE_DISCARD)
// ============================================================
void init_zero_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    for (PointInRectIterator<1> it(rect); it(); it++)
        acc[*it] = 0.0;
}

// ============================================================
// Task: Initialize a region from data passed as task argument
// Region 0: target (WRITE_DISCARD)
// Args: array of doubles
// ============================================================
void init_data_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const double *data = (const double *)task->args;
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    int idx = 0;
    for (PointInRectIterator<1> it(rect); it(); it++, idx++)
        acc[*it] = data[idx];
}

// ============================================================
// Task: scale_sum2  x = alpha1 * x + alpha2 * y
// Region 0: x (READ_WRITE)
// Region 1: y (READ_ONLY)
// Args: ScaleSum2Args
// ============================================================
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    ScaleSum2Args args = *(const ScaleSum2Args *)task->args;
    const FieldAccessor<READ_WRITE, double, 1> acc_x(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_y(regions[1], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    for (PointInRectIterator<1> it(rect); it(); it++)
        acc_x[*it] = args.alpha1 * acc_x[*it] + args.alpha2 * acc_y[*it];
}

// ============================================================
// Task: Extract first element of a subregion
// Region 0: source (READ_ONLY)
// Returns: double
// ============================================================
double extract_first_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.lo];
}

// ============================================================
// Task: Extract last element of a subregion
// Region 0: source (READ_ONLY)
// Returns: double
// ============================================================
double extract_last_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    return acc[rect.hi];
}

// ============================================================
// Task: System evaluation for the first block
// Region 0: q block (READ_ONLY)
// Region 1: dpdt block (WRITE_DISCARD)
// Future 0: right ghost (first element of next block)
// ============================================================
void system_first_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);
    double q_r = task->futures[0].get_result<double>();

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (coord_t i = lo; i <= hi; i++)
        q[i - lo] = acc_q[Point<1>(i)];

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(hi)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: System evaluation for a center block
// Region 0: q block (READ_ONLY)
// Region 1: dpdt block (WRITE_DISCARD)
// Future 0: left ghost (last element of previous block)
// Future 1: right ghost (first element of next block)
// ============================================================
void system_center_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);
    double q_l = task->futures[0].get_result<double>();
    double q_r = task->futures[1].get_result<double>();

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (coord_t i = lo; i <= hi; i++)
        q[i - lo] = acc_q[Point<1>(i)];

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(hi)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

// ============================================================
// Task: System evaluation for the last block
// Region 0: q block (READ_ONLY)
// Region 1: dpdt block (WRITE_DISCARD)
// Future 0: left ghost (last element of previous block)
// ============================================================
void system_last_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);
    double q_l = task->futures[0].get_result<double>();

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N);
    for (coord_t i = lo; i <= hi; i++)
        q[i - lo] = acc_q[Point<1>(i)];

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(lo + (coord_t)i)] = val;
    }
    acc_dpdt[Point<1>(hi)] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);
}

// ============================================================
// Task: Compute total energy
// Region 0: q (READ_ONLY), entire region
// Region 1: p (READ_ONLY), entire region
// Returns: double
// ============================================================
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_p(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    coord_t lo = rect.lo[0], hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    std::vector<double> q(N), p(N);
    for (coord_t i = lo; i <= hi; i++) {
        q[i - lo] = acc_q[Point<1>(i)];
        p[i - lo] = acc_p[Point<1>(i)];
    }

    double energy = 0.5 * checked_math::pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        energy += 0.5 * p[i] * p[i] + checked_math::pow(q[i], KAPPA) / KAPPA
            + checked_math::pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    energy += 0.5 * p[N - 1] * p[N - 1] + checked_math::pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return energy;
}

// ============================================================
// Helper: launch scale_sum2 on all M blocks
//   x_lp/x_lr: the region to update (RW)
//   y_lp/y_lr: the region to read (RO)
// ============================================================
void launch_scale_sum2(Context ctx, Runtime *runtime,
                       LogicalPartition x_lp, LogicalRegion x_lr,
                       LogicalPartition y_lp, LogicalRegion y_lr,
                       size_t M, double alpha1, double alpha2)
{
    ScaleSum2Args args;
    args.alpha1 = alpha1;
    args.alpha2 = alpha2;
    for (size_t i = 0; i < M; i++) {
        LogicalRegion x_sub = runtime->get_logical_subregion_by_color(x_lp, (coord_t)i);
        LogicalRegion y_sub = runtime->get_logical_subregion_by_color(y_lp, (coord_t)i);
        TaskLauncher launcher(SCALE_SUM2_TASK_ID,
            TaskArgument(&args, sizeof(args)));
        launcher.add_region_requirement(
            RegionRequirement(x_sub, READ_WRITE, EXCLUSIVE, x_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(y_sub, READ_ONLY, EXCLUSIVE, y_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

// ============================================================
// Helper: launch system evaluation (osc_chain) on all M blocks
//   Ghost values are communicated via Legion futures.
// ============================================================
void launch_system_eval(Context ctx, Runtime *runtime,
                        LogicalPartition q_lp, LogicalRegion q_lr,
                        LogicalPartition dpdt_lp, LogicalRegion dpdt_lr,
                        size_t M)
{
    // Extract boundary values from q blocks as futures
    // first_elem[i]: first element of block i (needed as right ghost for block i-1)
    // last_elem[i]: last element of block i (needed as left ghost for block i+1)
    std::vector<Future> first_elem(M);
    std::vector<Future> last_elem(M);

    for (size_t i = 0; i < M; i++) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(q_lp, (coord_t)i);
        if (i > 0) {
            // Extract first element (used as right ghost for block i-1)
            TaskLauncher launcher(EXTRACT_FIRST_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            first_elem[i] = runtime->execute_task(ctx, launcher);
        }
        if (i < M - 1) {
            // Extract last element (used as left ghost for block i+1)
            TaskLauncher launcher(EXTRACT_LAST_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            last_elem[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // Launch system computation tasks
    // First block
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(q_lp, (coord_t)0);
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(dpdt_lp, (coord_t)0);
        TaskLauncher launcher(SYSTEM_FIRST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(first_elem[1]); // right ghost
        runtime->execute_task(ctx, launcher);
    }

    // Middle blocks
    for (size_t i = 1; i < M - 1; i++) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(q_lp, (coord_t)i);
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(dpdt_lp, (coord_t)i);
        TaskLauncher launcher(SYSTEM_CENTER_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(last_elem[i - 1]);  // left ghost
        launcher.add_future(first_elem[i + 1]); // right ghost
        runtime->execute_task(ctx, launcher);
    }

    // Last block
    {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(q_lp, (coord_t)(M - 1));
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(dpdt_lp, (coord_t)(M - 1));
        TaskLauncher launcher(SYSTEM_LAST_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        launcher.add_future(last_elem[M - 2]); // left ghost
        runtime->execute_task(ctx, launcher);
    }
}

// ============================================================
// Top-level task
// ============================================================
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Parse command-line arguments
    const InputArgs &command_args = Runtime::get_input_args();
    size_t N = 1024;
    size_t G = 128;
    size_t steps = 100;
    double dt = 0.01;

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
    assert(M >= 2 && "Need at least 2 blocks (M >= 2)");
    assert(N == M * G && "N must be divisible by G");

    // Open output file
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt << std::endl;

    // Generate initial momentum data (same RNG as original)
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // --------------------------------------------------------
    // Create index space, field space, and logical regions
    // --------------------------------------------------------
    Rect<1> elem_rect(Point<1>(0), Point<1>((coord_t)(N - 1)));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
    runtime->attach_name(is, "element_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    runtime->attach_name(fs, "fields");

    LogicalRegion q_lr = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(q_lr, "q");
    runtime->attach_name(p_lr, "p");
    runtime->attach_name(dpdt_lr, "dpdt");

    // --------------------------------------------------------
    // Create equal partition into M blocks of size G
    // --------------------------------------------------------
    Rect<1> color_rect(Point<1>(0), Point<1>((coord_t)(M - 1)));
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);

    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp = runtime->get_logical_partition(q_lr, ip);
    LogicalPartition p_lp = runtime->get_logical_partition(p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(dpdt_lr, ip);

    // --------------------------------------------------------
    // Initialize q to zero, p from p_init
    // --------------------------------------------------------
    for (size_t i = 0; i < M; i++) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(q_lp, (coord_t)i);
        TaskLauncher launcher(INIT_ZERO_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, WRITE_DISCARD, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    for (size_t i = 0; i < M; i++) {
        LogicalRegion p_sub = runtime->get_logical_subregion_by_color(p_lp, (coord_t)i);
        TaskLauncher launcher(INIT_DATA_TASK_ID,
            TaskArgument(&p_init[i * G], G * sizeof(double)));
        launcher.add_region_requirement(
            RegionRequirement(p_sub, WRITE_DISCARD, EXCLUSIVE, p_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // --------------------------------------------------------
    // Compute initial energy
    // --------------------------------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // --------------------------------------------------------
    // Time integration: symplectic_rkn_sb3a_mclachlan
    //
    // Algorithm per step (from boost::odeint symplectic_nystroem_stepper_base):
    //   for l = 0 .. NUM_STAGES-1:
    //     q = q + COEF_A[l]*dt * p   // coordinate update
    //     sys(q, dpdt)               // compute force
    //     p = p + COEF_B[l]*dt * dpdt // momentum update
    // --------------------------------------------------------
    for (size_t step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {
            // Position update (every stage, including stage 0)
            launch_scale_sum2(ctx, runtime,
                              q_lp, q_lr, p_lp, p_lr,
                              M, 1.0, COEF_A[l] * dt);

            // System evaluation: dpdt = f(q)
            launch_system_eval(ctx, runtime,
                               q_lp, q_lr, dpdt_lp, dpdt_lr, M);

            // Momentum update
            if (COEF_B[l] != 0.0) {
                launch_scale_sum2(ctx, runtime,
                                  p_lp, p_lr, dpdt_lp, dpdt_lr,
                                  M, 1.0, COEF_B[l] * dt);
            }
        }
    }

    // --------------------------------------------------------
    // Compute final energy
    // --------------------------------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.region_requirements[1].add_field(FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ============================================================
// Main: register tasks and start runtime
// ============================================================
int main(int argc, char **argv)
{
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
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_FIRST_TASK_ID, "system_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_first_task>(registrar, "system_first");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_CENTER_TASK_ID, "system_center");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_center_task>(registrar, "system_center");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_LAST_TASK_ID, "system_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_last_task>(registrar, "system_last");
    }
    {
        TaskVariantRegistrar registrar(EXTRACT_FIRST_TASK_ID, "extract_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, extract_first_task>(registrar, "extract_first");
    }
    {
        TaskVariantRegistrar registrar(EXTRACT_LAST_TASK_ID, "extract_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, extract_last_task>(registrar, "extract_last");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
