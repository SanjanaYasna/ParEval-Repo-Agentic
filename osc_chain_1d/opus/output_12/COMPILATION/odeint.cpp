// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <functional>
#include <cstring>
#include <cassert>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

///////////////////////////////////////////////////////////////////////////////
// Physics constants (same as system.hpp)
///////////////////////////////////////////////////////////////////////////////
static const double KAPPA = 3.5;
static const double LAMBDA_EXP = 4.5;

namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

///////////////////////////////////////////////////////////////////////////////
// Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages, 4th order)
// Extracted from boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
///////////////////////////////////////////////////////////////////////////////
static const int NUM_STAGES = 6;
static const double COEF_A[NUM_STAGES] = {
     0.40518861839525227722,
    -0.28714404081652408900,
     0.38195542242127181178,
     0.38195542242127181178,
    -0.28714404081652408900,
     0.40518861839525227722
};

static const double W_B = 0.919661523017399857;
static const double X_B = 0.25 / W_B - 0.5 * W_B;
static const double Y_B = 1.0 - 2.0 * W_B - 2.0 * X_B;

static const double COEF_B[NUM_STAGES + 1] = {
    0.0, W_B, X_B, Y_B, X_B, W_B, 0.0
};

///////////////////////////////////////////////////////////////////////////////
// Legion IDs
///////////////////////////////////////////////////////////////////////////////
enum FieldIDs {
    FID_VAL = 101,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    GET_FIRST_TASK_ID,
    GET_LAST_TASK_ID,
    SYSTEM_TASK_ID,
    AXPY_TASK_ID,
    ENERGY_TASK_ID,
};

///////////////////////////////////////////////////////////////////////////////
// Argument structures
///////////////////////////////////////////////////////////////////////////////
struct SystemArgs {
    int block_idx;
    int num_blocks;
    int block_size;
};

struct AxpyArgs {
    double alpha;
};

///////////////////////////////////////////////////////////////////////////////
// GET_FIRST_TASK: returns first element of a subregion as Future<double>
///////////////////////////////////////////////////////////////////////////////
double get_first_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    return acc[rect.lo];
}

///////////////////////////////////////////////////////////////////////////////
// GET_LAST_TASK: returns last element of a subregion as Future<double>
///////////////////////////////////////////////////////////////////////////////
double get_last_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    return acc[rect.hi];
}

///////////////////////////////////////////////////////////////////////////////
// SYSTEM_TASK: compute dpdt for one block given q, boundary values, dpdt
// Uses local buffers to avoid read-after-write on WRITE_DISCARD accessor.
///////////////////////////////////////////////////////////////////////////////
void system_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    SystemArgs args;
    assert(task->arglen == sizeof(SystemArgs));
    memcpy(&args, task->args, sizeof(SystemArgs));

    int idx = args.block_idx;
    int M   = args.num_blocks;
    int G   = args.block_size;

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    Domain q_dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> q_rect = q_dom;
    coord_t start = q_rect.lo[0];

    // Copy q into a local buffer
    std::vector<double> q(G);
    for (int i = 0; i < G; ++i)
        q[i] = q_acc[start + i];

    // Retrieve boundary futures
    double q_l = 0.0; // left neighbor's rightmost value
    double q_r = 0.0; // right neighbor's leftmost value
    int fut_idx = 0;
    if (idx > 0) {
        q_l = task->futures[fut_idx].get_result<double>();
        fut_idx++;
    }
    if (idx < M - 1) {
        q_r = task->futures[fut_idx].get_result<double>();
        fut_idx++;
    }

    // Compute forces into local buffer
    std::vector<double> dpdt(G);

    double coupling_lr;
    if (idx == 0) {
        // first block: left wall coupling
        coupling_lr = -signed_pow(q[0], LAMBDA_EXP - 1);
    } else {
        coupling_lr = -signed_pow(q[0] - q_l, LAMBDA_EXP - 1);
    }

    for (int i = 0; i < G - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA_EXP - 1);
        dpdt[i] -= coupling_lr;
    }

    dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr;
    if (idx == M - 1) {
        // last block: right wall coupling
        dpdt[G - 1] -= signed_pow(q[G - 1], LAMBDA_EXP - 1);
    } else {
        dpdt[G - 1] -= signed_pow(q[G - 1] - q_r, LAMBDA_EXP - 1);
    }

    // Write dpdt back to region
    for (int i = 0; i < G; ++i)
        dpdt_acc[start + i] = dpdt[i];
}

///////////////////////////////////////////////////////////////////////////////
// AXPY_TASK: dest[i] += alpha * src[i]
///////////////////////////////////////////////////////////////////////////////
void axpy_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
    AxpyArgs args;
    assert(task->arglen == sizeof(AxpyArgs));
    memcpy(&args, task->args, sizeof(AxpyArgs));

    const FieldAccessor<READ_WRITE, double, 1> dest_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  src_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    for (PointInRectIterator<1> itr(rect); itr(); itr++) {
        dest_acc[*itr] = dest_acc[*itr] + args.alpha * src_acc[*itr];
    }
}

///////////////////////////////////////////////////////////////////////////////
// ENERGY_TASK: compute total energy (serial, reads all of q and p)
///////////////////////////////////////////////////////////////////////////////
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];
    size_t N = (size_t)(hi - lo + 1);

    // Copy to local buffers
    std::vector<double> q(N), p(N);
    for (size_t i = 0; i < N; ++i) {
        q[i] = q_acc[lo + (coord_t)i];
        p[i] = p_acc[lo + (coord_t)i];
    }

    using checked_math::pow;

    double en = 0.5 * pow(std::abs(q[0]), LAMBDA_EXP) / LAMBDA_EXP;
    for (size_t i = 0; i < N - 1; ++i) {
        en += 0.5 * p[i] * p[i]
            + pow(q[i], KAPPA) / KAPPA
            + pow(std::abs(q[i] - q[i + 1]), LAMBDA_EXP) / LAMBDA_EXP;
    }
    en += 0.5 * p[N - 1] * p[N - 1]
        + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(std::abs(q[N - 1]), LAMBDA_EXP) / LAMBDA_EXP;

    return en;
}

///////////////////////////////////////////////////////////////////////////////
// Helper: launch system evaluation for all blocks
///////////////////////////////////////////////////////////////////////////////
static void launch_system(Context ctx, Runtime *runtime,
                          LogicalRegion q_lr, LogicalRegion dpdt_lr,
                          LogicalPartition q_lp, LogicalPartition dpdt_lp,
                          IndexSpace color_is, int M, int G)
{
    // First, launch boundary extraction tasks
    std::vector<Future> f_last(M);   // last element of each q block
    std::vector<Future> f_first(M);  // first element of each q block

    for (int i = 0; i < M; ++i) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(ctx, q_lp, i);

        {
            TaskLauncher launcher(GET_LAST_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            f_last[i] = runtime->execute_task(ctx, launcher);
        }
        {
            TaskLauncher launcher(GET_FIRST_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            f_first[i] = runtime->execute_task(ctx, launcher);
        }
    }

    // Launch system tasks
    for (int i = 0; i < M; ++i) {
        LogicalRegion q_sub    = runtime->get_logical_subregion_by_color(ctx, q_lp, i);
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(ctx, dpdt_lp, i);

        SystemArgs args;
        args.block_idx   = i;
        args.num_blocks  = M;
        args.block_size  = G;

        TaskLauncher launcher(SYSTEM_TASK_ID,
            TaskArgument(&args, sizeof(SystemArgs)));

        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        // Add boundary futures
        if (i > 0)
            launcher.add_future(f_last[i - 1]);   // left neighbor's last element
        if (i < M - 1)
            launcher.add_future(f_first[i + 1]);   // right neighbor's first element

        runtime->execute_task(ctx, launcher);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Helper: launch axpy (dest += alpha * src) on all blocks
///////////////////////////////////////////////////////////////////////////////
static void launch_axpy(Context ctx, Runtime *runtime,
                        LogicalRegion dest_lr, LogicalRegion src_lr,
                        LogicalPartition dest_lp, LogicalPartition src_lp,
                        IndexSpace color_is, int M, double alpha)
{
    AxpyArgs args;
    args.alpha = alpha;

    for (int i = 0; i < M; ++i) {
        LogicalRegion dest_sub = runtime->get_logical_subregion_by_color(ctx, dest_lp, i);
        LogicalRegion src_sub  = runtime->get_logical_subregion_by_color(ctx, src_lp, i);

        TaskLauncher launcher(AXPY_TASK_ID,
            TaskArgument(&args, sizeof(AxpyArgs)));

        launcher.add_region_requirement(
            RegionRequirement(dest_sub, READ_WRITE, EXCLUSIVE, dest_lr));
        launcher.region_requirements[0].add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(src_sub, READ_ONLY, EXCLUSIVE, src_lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Helper: compute energy
///////////////////////////////////////////////////////////////////////////////
static double compute_energy(Context ctx, Runtime *runtime,
                             LogicalRegion q_lr, LogicalRegion p_lr)
{
    TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
    launcher.add_region_requirement(
        RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    launcher.region_requirements[0].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    launcher.region_requirements[1].add_field(FID_VAL);

    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

///////////////////////////////////////////////////////////////////////////////
// Command-line parsing
///////////////////////////////////////////////////////////////////////////////
struct AppConfig {
    std::size_t N;
    std::size_t G;
    std::size_t steps;
    double dt;
};

static AppConfig parse_config(int argc, char **argv)
{
    AppConfig cfg;
    cfg.N     = 1024;
    cfg.G     = 128;
    cfg.steps = 100;
    cfg.dt    = 0.01;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--N") == 0 && i + 1 < argc)
            cfg.N = (std::size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--G") == 0 && i + 1 < argc)
            cfg.G = (std::size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            cfg.steps = (std::size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
            cfg.dt = atof(argv[++i]);
    }
    return cfg;
}

///////////////////////////////////////////////////////////////////////////////
// TOP LEVEL TASK
///////////////////////////////////////////////////////////////////////////////
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Parse command-line arguments
    const InputArgs &command_args = Runtime::get_input_args();
    AppConfig cfg = parse_config(command_args.argc, command_args.argv);

    const std::size_t N = cfg.N;
    const std::size_t G = cfg.G;
    const std::size_t steps = cfg.steps;
    const double dt = cfg.dt;
    const std::size_t M = N / G;

    // Open output file
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G;
    outfile << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt << std::endl;

    // Generate random initial momenta (same seed as HPX version)
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // Create index spaces
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, (coord_t)(N - 1)));
    IndexSpace color_is = runtime->create_index_space(ctx, Rect<1>(0, (coord_t)(M - 1)));

    // Create field space
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Create logical regions for q, p, dpdt
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // Create equal partitions
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // Initialize q to zeros and p to random values using inline mappings
    {
        RegionRequirement q_req(q_lr, WRITE_DISCARD, EXCLUSIVE, q_lr);
        q_req.add_field(FID_VAL);
        InlineLauncher q_il(q_req);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
        q_pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> q_acc(q_pr, FID_VAL);
        for (std::size_t i = 0; i < N; ++i)
            q_acc[i] = 0.0;
        runtime->unmap_region(ctx, q_pr);
    }
    {
        RegionRequirement p_req(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr);
        p_req.add_field(FID_VAL);
        InlineLauncher p_il(p_req);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
        p_pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> p_acc(p_pr, FID_VAL);
        for (std::size_t i = 0; i < N; ++i)
            p_acc[i] = p_init[i];
        runtime->unmap_region(ctx, p_pr);
    }
    // Initialize dpdt to zeros
    {
        RegionRequirement dpdt_req(dpdt_lr, WRITE_DISCARD, EXCLUSIVE, dpdt_lr);
        dpdt_req.add_field(FID_VAL);
        InlineLauncher dpdt_il(dpdt_req);
        PhysicalRegion dpdt_pr = runtime->map_region(ctx, dpdt_il);
        dpdt_pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(dpdt_pr, FID_VAL);
        for (std::size_t i = 0; i < N; ++i)
            dpdt_acc[i] = 0.0;
        runtime->unmap_region(ctx, dpdt_pr);
    }

    // Compute initial energy
    double e_init = compute_energy(ctx, runtime, q_lr, p_lr);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(e_init)) << std::endl;

    // Integration loop: symplectic_rkn_sb3a_mclachlan stepper
    // Algorithm per step:
    //   for l = 0..NUM_STAGES-1:
    //     q += dt * COEF_B[l] * p       (skip if l==0 since COEF_B[0]==0)
    //     dpdt = f(q)                    (system evaluation)
    //     p += dt * COEF_A[l] * dpdt
    //   q += dt * COEF_B[NUM_STAGES] * p  (skip since COEF_B[6]==0)
    for (std::size_t step = 0; step < steps; ++step) {
        for (int l = 0; l < NUM_STAGES; ++l) {
            // Position update (skip for l == 0 since COEF_B[0] == 0)
            if (l > 0 && COEF_B[l] != 0.0) {
                launch_axpy(ctx, runtime, q_lr, p_lr, q_lp, p_lp,
                            color_is, (int)M, dt * COEF_B[l]);
            }

            // System evaluation: dpdt = f(q)
            launch_system(ctx, runtime, q_lr, dpdt_lr, q_lp, dpdt_lp,
                          color_is, (int)M, (int)G);

            // Momentum update: p += dt * COEF_A[l] * dpdt
            launch_axpy(ctx, runtime, p_lr, dpdt_lr, p_lp, dpdt_lp,
                        color_is, (int)M, dt * COEF_A[l]);
        }

        // Final position update: q += dt * COEF_B[NUM_STAGES] * p
        if (COEF_B[NUM_STAGES] != 0.0) {
            launch_axpy(ctx, runtime, q_lr, p_lr, q_lp, p_lp,
                        color_is, (int)M, dt * COEF_B[NUM_STAGES]);
        }
    }

    // Compute final energy
    double e_final = compute_energy(ctx, runtime, q_lr, p_lr);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(e_final)) << std::endl;

    outfile.close();

    // Clean up
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

///////////////////////////////////////////////////////////////////////////////
// MAIN
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(GET_FIRST_TASK_ID, "get_first");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, get_first_task>(registrar, "get_first");
    }
    {
        TaskVariantRegistrar registrar(GET_LAST_TASK_ID, "get_last");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, get_last_task>(registrar, "get_last");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_TASK_ID, "system");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_task>(registrar, "system");
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
