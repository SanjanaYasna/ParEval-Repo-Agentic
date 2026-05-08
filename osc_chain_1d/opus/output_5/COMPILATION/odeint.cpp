// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <cmath>
#include <functional>
#include <cstring>
#include <cstdlib>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ---------------------------------------------------------------------------
// Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages, order 4)
// Matches boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
// ---------------------------------------------------------------------------
static const int NUM_STAGES = 6;

static const double COEF_A_0 =  0.40518861839525227722;
static const double COEF_A_1 = -0.28714404081652408900;
static const double COEF_A_2 =  0.5 - COEF_A_0 - COEF_A_1;

static const double COEF_A[NUM_STAGES] = {
    COEF_A_0, COEF_A_1, COEF_A_2, COEF_A_2, COEF_A_1, COEF_A_0
};

static const double COEF_B_0 = -3.0 / 73.0;
static const double COEF_B_1 =  17.0 / 59.0;
static const double COEF_B_2 =  1.0 - 2.0 * (COEF_B_0 + COEF_B_1);

static const double COEF_B[NUM_STAGES] = {
    COEF_B_0, COEF_B_1, COEF_B_2, COEF_B_1, COEF_B_0, 0.0
};

// ---------------------------------------------------------------------------
// Task IDs
// ---------------------------------------------------------------------------
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    INIT_ZERO_TASK_ID,
    INIT_COPY_TASK_ID,
    SYSTEM_TASK_ID,
    SCALE_SUM2_TASK_ID,
    GET_FIRST_ELEM_TASK_ID,
    GET_LAST_ELEM_TASK_ID,
};

// ---------------------------------------------------------------------------
// Field IDs
// ---------------------------------------------------------------------------
enum FieldIDs {
    FID_VAL = 0,
};

// ---------------------------------------------------------------------------
// Math helpers (from system.hpp)
// ---------------------------------------------------------------------------
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

// ---------------------------------------------------------------------------
// Task argument structures
// ---------------------------------------------------------------------------
struct SystemArgs {
    int block_idx;
    int num_blocks;
};

struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

// ---------------------------------------------------------------------------
// INIT_ZERO_TASK – fill a sub-region with zeros
// ---------------------------------------------------------------------------
void init_zero_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc[*pir] = 0.0;
}

// ---------------------------------------------------------------------------
// INIT_COPY_TASK – copy data from task argument into a sub-region
// ---------------------------------------------------------------------------
void init_copy_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const double *data = reinterpret_cast<const double *>(task->args);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    int idx = 0;
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc[*pir] = data[idx++];
}

// ---------------------------------------------------------------------------
// GET_FIRST_ELEM_TASK – return the first element of a sub-region
// ---------------------------------------------------------------------------
double get_first_elem_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    return acc[rect.lo];
}

// ---------------------------------------------------------------------------
// GET_LAST_ELEM_TASK – return the last element of a sub-region
// ---------------------------------------------------------------------------
double get_last_elem_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    return acc[rect.hi];
}

// ---------------------------------------------------------------------------
// SYSTEM_TASK – compute dpdt = f(q) for one block
// Region 0: Q block (READ_ONLY)
// Region 1: DPDT block (WRITE_DISCARD)
// Futures: left ghost (if not first), right ghost (if not last)
// ---------------------------------------------------------------------------
void system_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const SystemArgs *args = reinterpret_cast<const SystemArgs *>(task->args);
    const int block_idx  = args->block_idx;
    const int num_blocks = args->num_blocks;
    const bool is_first  = (block_idx == 0);
    const bool is_last   = (block_idx == num_blocks - 1);

    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    const coord_t lo = rect.lo[0];
    const size_t  N  = rect.hi[0] - lo + 1;

    // Read q into a local buffer
    std::vector<double> q(N);
    for (size_t i = 0; i < N; i++)
        q[i] = acc_q[lo + (coord_t)i];

    // Retrieve ghost values from futures
    int fidx = 0;
    double q_left_ghost  = 0.0;
    double q_right_ghost = 0.0;
    if (!is_first)
        q_left_ghost = task->futures[fidx++].get_result<double>();
    if (!is_last)
        q_right_ghost = task->futures[fidx++].get_result<double>();

    // Compute forces (matches system.hpp logic)
    std::vector<double> dpdt(N);

    double coupling_lr;
    if (is_first)
        coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    else
        coupling_lr = -signed_pow(q[0] - q_left_ghost, LAMBDA - 1);

    for (size_t i = 0; i < N - 1; i++) {
        dpdt[i]     = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr =  signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i]    -= coupling_lr;
    }

    if (is_last) {
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                      + coupling_lr
                      - signed_pow(q[N - 1], LAMBDA - 1);
    } else {
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                      + coupling_lr
                      - signed_pow(q[N - 1] - q_right_ghost, LAMBDA - 1);
    }

    // Write dpdt
    for (size_t i = 0; i < N; i++)
        acc_dpdt[lo + (coord_t)i] = dpdt[i];
}

// ---------------------------------------------------------------------------
// SCALE_SUM2_TASK – x = alpha1*x + alpha2*y
// Region 0: x (READ_WRITE)
// Region 1: y (READ_ONLY)
// ---------------------------------------------------------------------------
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const ScaleSum2Args *args =
        reinterpret_cast<const ScaleSum2Args *>(task->args);

    const FieldAccessor<READ_WRITE, double, 1> acc_x(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> acc_y(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc_x[*pir] = args->alpha1 * acc_x[*pir] + args->alpha2 * acc_y[*pir];
}

// ---------------------------------------------------------------------------
// Helper: launch system evaluation across all M blocks
// ---------------------------------------------------------------------------
static void launch_system_eval(Context ctx, Runtime *runtime,
                               LogicalPartition q_lp,  LogicalRegion q_lr,
                               LogicalPartition dpdt_lp, LogicalRegion dpdt_lr,
                               size_t M)
{
    // 1. Extract boundary values from every block
    std::vector<Future> first_elems(M);
    std::vector<Future> last_elems(M);

    for (size_t i = 0; i < M; i++) {
        LogicalRegion q_sub =
            runtime->get_logical_subregion_by_color(ctx, q_lp, (coord_t)i);

        {
            TaskLauncher launcher(GET_FIRST_ELEM_TASK_ID,
                                  TaskArgument(nullptr, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            first_elems[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(GET_LAST_ELEM_TASK_ID,
                                  TaskArgument(nullptr, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            last_elems[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // 2. Launch system tasks with ghost-value futures
    for (size_t i = 0; i < M; i++) {
        LogicalRegion q_sub =
            runtime->get_logical_subregion_by_color(ctx, q_lp, (coord_t)i);
        LogicalRegion dpdt_sub =
            runtime->get_logical_subregion_by_color(ctx, dpdt_lp, (coord_t)i);

        SystemArgs sargs;
        sargs.block_idx  = (int)i;
        sargs.num_blocks = (int)M;

        TaskLauncher launcher(SYSTEM_TASK_ID,
                              TaskArgument(&sargs, sizeof(sargs)));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        // Left ghost (last element of block i-1)
        if (i > 0)
            launcher.add_future(last_elems[i - 1]);
        // Right ghost (first element of block i+1)
        if (i < M - 1)
            launcher.add_future(first_elems[i + 1]);

        runtime->execute_task(ctx, launcher);
    }
}

// ---------------------------------------------------------------------------
// Helper: launch scale_sum2 across all M blocks
//   rw_region = alpha1 * rw_region + alpha2 * ro_region
// ---------------------------------------------------------------------------
static void launch_scale_sum2(Context ctx, Runtime *runtime,
                              LogicalPartition rw_lp, LogicalRegion rw_lr,
                              LogicalPartition ro_lp, LogicalRegion ro_lr,
                              size_t M, double alpha1, double alpha2)
{
    ScaleSum2Args sargs;
    sargs.alpha1 = alpha1;
    sargs.alpha2 = alpha2;

    for (size_t i = 0; i < M; i++) {
        LogicalRegion rw_sub =
            runtime->get_logical_subregion_by_color(ctx, rw_lp, (coord_t)i);
        LogicalRegion ro_sub =
            runtime->get_logical_subregion_by_color(ctx, ro_lp, (coord_t)i);

        TaskLauncher launcher(SCALE_SUM2_TASK_ID,
                              TaskArgument(&sargs, sizeof(sargs)));
        launcher.add_region_requirement(
            RegionRequirement(rw_sub, READ_WRITE, EXCLUSIVE, rw_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(ro_sub, READ_ONLY, EXCLUSIVE, ro_lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }
}

// ---------------------------------------------------------------------------
// Energy computation (inline-mapped, sequential – mirrors system.hpp energy())
// ---------------------------------------------------------------------------
static double compute_energy(Context ctx, Runtime *runtime,
                             LogicalRegion q_lr, LogicalRegion p_lr,
                             size_t N)
{
    RegionRequirement req_q(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
    req_q.add_field(FID_VAL);
    RegionRequirement req_p(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
    req_p.add_field(FID_VAL);

    InlineLauncher il_q(req_q);
    InlineLauncher il_p(req_p);

    PhysicalRegion pr_q = runtime->map_region(ctx, il_q);
    PhysicalRegion pr_p = runtime->map_region(ctx, il_p);
    pr_q.wait_until_valid();
    pr_p.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> acc_q(pr_q, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_p(pr_p, FID_VAL);

    std::vector<double> q(N), p(N);
    for (size_t i = 0; i < N; i++) {
        q[i] = acc_q[(coord_t)i];
        p[i] = acc_p[(coord_t)i];
    }

    runtime->unmap_region(ctx, pr_q);
    runtime->unmap_region(ctx, pr_p);

    // Same formula as in system.hpp energy(dvec,dvec)
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

// ---------------------------------------------------------------------------
// Top-level task
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing (--N, --G, --steps, --dt) -------------------
    size_t N     = 1024;
    size_t G     = 128;
    size_t steps = 100;
    double dt    = 0.01;

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

    // ---- output file ------------------------------------------------------
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

    // ---- create index space & field space ---------------------------------
    Rect<1> elem_rect(0, (coord_t)(N - 1));
    IndexSpace is = runtime->create_index_space(ctx, elem_rect);

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // ---- create logical regions for q, p, dpdt ----------------------------
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // ---- equal partition into M blocks of G elements ----------------------
    Rect<1> color_rect(0, (coord_t)(M - 1));
    IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // ---- generate random initial momentum (same RNG as HPX version) -------
    std::vector<double> p_init(N);
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));
    }

    // ---- initialize q = 0, p = p_init ------------------------------------
    for (size_t i = 0; i < M; i++) {
        // q block → zero
        {
            LogicalRegion q_sub =
                runtime->get_logical_subregion_by_color(ctx, q_lp, (coord_t)i);
            TaskLauncher launcher(INIT_ZERO_TASK_ID,
                                  TaskArgument(nullptr, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, WRITE_DISCARD, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // p block → copy from p_init
        {
            LogicalRegion p_sub =
                runtime->get_logical_subregion_by_color(ctx, p_lp, (coord_t)i);
            TaskLauncher launcher(INIT_COPY_TASK_ID,
                TaskArgument(&p_init[i * G], G * sizeof(double)));
            launcher.add_region_requirement(
                RegionRequirement(p_sub, WRITE_DISCARD, EXCLUSIVE, p_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // dpdt block → zero (will be overwritten before use)
        {
            LogicalRegion dpdt_sub =
                runtime->get_logical_subregion_by_color(ctx, dpdt_lp, (coord_t)i);
            TaskLauncher launcher(INIT_ZERO_TASK_ID,
                                  TaskArgument(nullptr, 0));
            launcher.add_region_requirement(
                RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }

    // ---- compute initial energy -------------------------------------------
    double e0 = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(e0)) << std::endl;

    // ---- symplectic_rkn_sb3a_mclachlan integration ------------------------
    // Matches the boost::numeric::odeint implementation exactly:
    //   do_step_impl: for l=0..NUM_STAGES-1:
    //       if l > 0: dpdt = f(q)
    //       p += b[l]*dt*dpdt
    //       q += a[l]*dt*p
    //   After loop (in do_step wrapper for next call): dpdt = f(q)
    //
    // The initial dpdt = f(q) is computed once before the first step.
    // Between steps, dpdt = f(q) is recomputed (FSAL).

    // Initial force evaluation (corresponds to sys(q, dpdt) at start of first do_step)
    launch_system_eval(ctx, runtime, q_lp, q_lr, dpdt_lp, dpdt_lr, M);

    for (size_t step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {
            // For l > 0, recompute dpdt = f(q) BEFORE the momentum update
            if (l > 0) {
                launch_system_eval(ctx, runtime,
                                   q_lp, q_lr,
                                   dpdt_lp, dpdt_lr, M);
            }

            // p = 1.0 * p + b[l]*dt * dpdt
            launch_scale_sum2(ctx, runtime,
                              p_lp, p_lr,
                              dpdt_lp, dpdt_lr,
                              M, 1.0, COEF_B[l] * dt);

            // q = 1.0 * q + a[l]*dt * p
            launch_scale_sum2(ctx, runtime,
                              q_lp, q_lr,
                              p_lp, p_lr,
                              M, 1.0, COEF_A[l] * dt);
        }
        // After the loop: recompute dpdt = f(q) for next step's l=0
        // (FSAL: this is the initial force eval for the next do_step call)
        // We always do this, even after the last step, to match boost exactly.
        // (b[5]=0 means the extra p-update is a no-op anyway.)
        launch_system_eval(ctx, runtime,
                           q_lp, q_lr,
                           dpdt_lp, dpdt_lr, M);
    }

    // ---- compute final energy ---------------------------------------------
    double e1 = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(e1)) << std::endl;

    outfile.close();

    // ---- cleanup ----------------------------------------------------------
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
// ---------------------------------------------------------------------------
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
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<init_zero_task>(registrar, "init_zero");
    }
    {
        TaskVariantRegistrar registrar(INIT_COPY_TASK_ID, "init_copy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<init_copy_task>(registrar, "init_copy");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_TASK_ID, "system");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_task>(registrar, "system");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
    {
        TaskVariantRegistrar registrar(GET_FIRST_ELEM_TASK_ID, "get_first_elem");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, get_first_elem_task>(
            registrar, "get_first_elem");
    }
    {
        TaskVariantRegistrar registrar(GET_LAST_ELEM_TASK_ID, "get_last_elem");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, get_last_elem_task>(
            registrar, "get_last_elem");
    }

    return Runtime::start(argc, argv);
}
