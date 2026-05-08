// Legion translation of HPX oscillator chain ODEINT
// Implements symplectic_rkn_sb3a_mclachlan integration of 1D nonlinear oscillator chain

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <cstring>
#include <functional>

#include <legion.h>
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// Physical constants
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// Field and Task IDs
enum FieldIDs {
    FID_VAL = 0,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_TASK_ID,
    ENERGY_TASK_ID,
};

// Math helpers (matching system.hpp)
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

// Task argument structs
struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

struct SystemArgs {
    int num_blocks;
    int block_idx;
    int G;
    int N;
};

// ---------------------------------------------------------------------------
// scale_sum2 task: x1[i] = alpha1 * x1[i] + alpha2 * x3[i]
//   Region 0: x1 (READ_WRITE) — the array being updated (also serves as x2)
//   Region 1: x3 (READ_ONLY)  — the second source array
// ---------------------------------------------------------------------------
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    ScaleSum2Args args = *(const ScaleSum2Args *)task->args;

    const FieldAccessor<READ_WRITE, double, 1> acc_x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> acc_x3(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        acc_x1[*pir] = args.alpha1 * acc_x1[*pir] + args.alpha2 * acc_x3[*pir];
    }
}

// ---------------------------------------------------------------------------
// system task: compute dpdt = f(q) for one block
//   Region 0: full q  (READ_ONLY)
//   Region 1: dpdt partition (WRITE_DISCARD)
// ---------------------------------------------------------------------------
void system_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime) {
    SystemArgs args = *(const SystemArgs *)task->args;
    int block_idx = args.block_idx;
    int M = args.num_blocks;

    const FieldAccessor<READ_ONLY,     double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    // Derive actual block boundaries from the sub-region
    Domain dom = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());
    Rect<1> rect(dom);
    int start = rect.lo[0];
    int end   = rect.hi[0] + 1;  // exclusive

    bool is_first = (block_idx == 0);
    bool is_last  = (block_idx == M - 1);

    // Left boundary coupling
    double coupling_lr;
    if (is_first) {
        // Fixed left boundary (q_{-1} = 0)
        coupling_lr = -signed_pow(acc_q[Point<1>(start)], LAMBDA - 1);
    } else {
        double q_l = acc_q[Point<1>(start - 1)];
        coupling_lr = -signed_pow(acc_q[Point<1>(start)] - q_l, LAMBDA - 1);
    }

    // Interior of block
    for (int i = start; i < end - 1; i++) {
        double val = -signed_pow(acc_q[Point<1>(i)], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(acc_q[Point<1>(i)] - acc_q[Point<1>(i + 1)], LAMBDA - 1);
        val -= coupling_lr;
        acc_dpdt[Point<1>(i)] = val;
    }

    // Last element of block — right boundary coupling
    if (is_last) {
        // Fixed right boundary (q_{N} = 0)
        acc_dpdt[Point<1>(end - 1)] =
            -signed_pow(acc_q[Point<1>(end - 1)], KAPPA - 1)
            + coupling_lr
            - signed_pow(acc_q[Point<1>(end - 1)], LAMBDA - 1);
    } else {
        double q_r = acc_q[Point<1>(end)];
        acc_dpdt[Point<1>(end - 1)] =
            -signed_pow(acc_q[Point<1>(end - 1)], KAPPA - 1)
            + coupling_lr
            - signed_pow(acc_q[Point<1>(end - 1)] - q_r, LAMBDA - 1);
    }
}

// ---------------------------------------------------------------------------
// energy task: compute total energy of the chain
//   Region 0: q (READ_ONLY)
//   Region 1: p (READ_ONLY)
// ---------------------------------------------------------------------------
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
    int N = *(const int *)task->args;

    const FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_p(regions[1], FID_VAL);

    // Left boundary potential
    double e = 0.5 * checked_math::pow(acc_q[Point<1>(0)], LAMBDA) / LAMBDA;

    for (int i = 0; i < N - 1; i++) {
        double qi  = acc_q[Point<1>(i)];
        double qi1 = acc_q[Point<1>(i + 1)];
        double pi  = acc_p[Point<1>(i)];
        e += 0.5 * pi * pi
           + checked_math::pow(qi, KAPPA) / KAPPA
           + checked_math::pow(qi - qi1, LAMBDA) / LAMBDA;
    }

    double qN1 = acc_q[Point<1>(N - 1)];
    double pN1 = acc_p[Point<1>(N - 1)];
    e += 0.5 * pN1 * pN1
       + checked_math::pow(qN1, KAPPA) / KAPPA
       + 0.5 * checked_math::pow(qN1, LAMBDA) / LAMBDA;

    return e;
}

// ---------------------------------------------------------------------------
// Top-level task
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // ---- Default parameters ------------------------------------------------
    int    N     = 1024;
    int    G     = 128;
    int    steps = 100;
    double dt    = 0.01;

    // ---- Parse command-line arguments --------------------------------------
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--N") == 0 && i + 1 < command_args.argc)
            N = atoi(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--G") == 0 && i + 1 < command_args.argc)
            G = atoi(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--steps") == 0 && i + 1 < command_args.argc)
            steps = atoi(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--dt") == 0 && i + 1 < command_args.argc)
            dt = atof(command_args.argv[++i]);
    }

    int M = N / G;

    // ---- Output file -------------------------------------------------------
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

    // ---- Generate initial momenta (same RNG as original) -------------------
    std::vector<double> p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    // ---- Create index space and field space --------------------------------
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // ---- Create logical regions for q, p, dpdt ----------------------------
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // ---- Create equal partition into M blocks ------------------------------
    IndexSpace     color_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));
    IndexPartition ip       = runtime->create_equal_partition(ctx, is, color_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // ---- Initialize q to zero ----------------------------------------------
    runtime->fill_field(ctx, q_lr, q_lr, FID_VAL, 0.0);

    // ---- Initialize p with random values via inline mapping ----------------
    {
        InlineLauncher il(RegionRequirement(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr));
        il.add_field(FID_VAL);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++)
            acc[Point<1>(i)] = p_init[i];
        runtime->unmap_region(ctx, pr);
    }

    // ---- Initialize dpdt to zero -------------------------------------------
    runtime->fill_field(ctx, dpdt_lr, dpdt_lr, FID_VAL, 0.0);

    // ---- Compute and print initial energy ----------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(&N, sizeof(N)));
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

    // ---- Symplectic RKN SB3A McLachlan coefficients (6 stages) -------------
    const int NUM_STAGES = 6;
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

    Domain launch_domain(Rect<1>(0, M - 1));

    // ---- Time integration loop ---------------------------------------------
    for (int step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {

            // --- Drift: q = 1.0 * q + coef_a[l]*dt * p ---------------------
            {
                ScaleSum2Args ss_args;
                ss_args.alpha1 = 1.0;
                ss_args.alpha2 = coef_a[l] * dt;

                IndexLauncher launcher(SCALE_SUM2_TASK_ID, launch_domain,
                    TaskArgument(&ss_args, sizeof(ss_args)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_lp, 0, READ_WRITE, EXCLUSIVE, q_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_ONLY, EXCLUSIVE, p_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // --- Evaluate system: dpdt = f(q) --------------------------------
            // Use individual TaskLaunchers so that each task reads the full q
            // region with EXCLUSIVE coherence. Multiple READ_ONLY+EXCLUSIVE
            // tasks on the same region can execute concurrently (no read-read
            // conflict), and EXCLUSIVE coherence properly enforces RAW
            // dependencies with the preceding drift that wrote to q partitions.
            {
                for (int blk = 0; blk < M; blk++) {
                    SystemArgs sys_args;
                    sys_args.num_blocks = M;
                    sys_args.block_idx  = blk;
                    sys_args.G          = G;
                    sys_args.N          = N;

                    LogicalRegion dpdt_sub =
                        runtime->get_logical_subregion_by_color(
                            dpdt_lp, DomainPoint(Point<1>(blk)));

                    TaskLauncher launcher(SYSTEM_TASK_ID,
                        TaskArgument(&sys_args, sizeof(sys_args)));
                    // Region 0: full q — READ_ONLY, EXCLUSIVE
                    launcher.add_region_requirement(
                        RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
                    launcher.region_requirements[0].add_field(FID_VAL);
                    // Region 1: dpdt sub-region — WRITE_DISCARD, EXCLUSIVE
                    launcher.add_region_requirement(
                        RegionRequirement(dpdt_sub, WRITE_DISCARD,
                                          EXCLUSIVE, dpdt_lr));
                    launcher.region_requirements[1].add_field(FID_VAL);
                    runtime->execute_task(ctx, launcher);
                }
            }

            // --- Kick: p = 1.0 * p + coef_b[l]*dt * dpdt -------------------
            {
                ScaleSum2Args ss_args;
                ss_args.alpha1 = 1.0;
                ss_args.alpha2 = coef_b[l] * dt;

                IndexLauncher launcher(SCALE_SUM2_TASK_ID, launch_domain,
                    TaskArgument(&ss_args, sizeof(ss_args)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_WRITE, EXCLUSIVE, p_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0, READ_ONLY, EXCLUSIVE, dpdt_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }
        }
    }

    // ---- Compute and print final energy ------------------------------------
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(&N, sizeof(N)));
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

    // ---- Cleanup -----------------------------------------------------------
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ---------------------------------------------------------------------------
// main — register tasks and start the Legion runtime
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_TASK_ID, "system");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<system_task>(registrar, "system");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
