// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>

#include "legion.h"

using namespace Legion;

// ============================================================
// Physics constants (from system.hpp)
// ============================================================
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ============================================================
// Symplectic RKN SB3A McLachlan stepper coefficients
// (from boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan)
// ============================================================
static const int NUM_STAGES = 6;

static const double coef_a[NUM_STAGES] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
    0.5 - 0.40518861839525227722 + 0.28714404081652408900,
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

// ============================================================
// Task and Field IDs
// ============================================================
enum {
    TOP_LEVEL_TASK_ID,
    FORCE_TASK_ID,
    AXPY_TASK_ID,
};

enum {
    FID_VAL = 0,
};

// ============================================================
// Physics helper functions (from system.hpp)
// ============================================================
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow(x, k) * s;
}

// ============================================================
// AXPY task arguments: x = alpha1*x + alpha2*y
// (replaces shared_operations.hpp scale_sum2)
// ============================================================
struct AxpyArgs {
    double alpha1;
    double alpha2;
};

// ============================================================
// Force computation task
// Replaces osc_chain / system_first_block / system_center_block / system_last_block
// regions[0]: q with ghost cells (READ_ONLY, from aliased ghost partition)
// regions[1]: dpdt (WRITE_DISCARD, from disjoint partition)
// ============================================================
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    // Determine own range from the dpdt (disjoint) region
    Rect<1> own_rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    coord_t own_lo = own_rect.lo[0];
    coord_t own_hi = own_rect.hi[0];

    // Determine ghost range from the q (ghost) region
    Rect<1> q_rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    bool has_left  = (q_rect.lo[0] < own_lo);
    bool has_right = (q_rect.hi[0] > own_hi);

    // Compute initial left coupling
    double coupling_lr;
    if (!has_left) {
        // First block: boundary coupling with fixed wall at 0
        coupling_lr = -signed_pow(q_acc[own_lo], LAMBDA - 1);
    } else {
        coupling_lr = -signed_pow(q_acc[own_lo] - q_acc[own_lo - 1], LAMBDA - 1);
    }

    // Interior elements
    for (coord_t i = own_lo; i < own_hi; ++i) {
        double val = -signed_pow(q_acc[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q_acc[i] - q_acc[i + 1], LAMBDA - 1);
        val -= coupling_lr;
        dpdt_acc[i] = val;
    }

    // Last element of this block
    double val = -signed_pow(q_acc[own_hi], KAPPA - 1) + coupling_lr;
    if (!has_right) {
        // Last block: boundary coupling with fixed wall at 0
        val -= signed_pow(q_acc[own_hi], LAMBDA - 1);
    } else {
        val -= signed_pow(q_acc[own_hi] - q_acc[own_hi + 1], LAMBDA - 1);
    }
    dpdt_acc[own_hi] = val;
}

// ============================================================
// AXPY task: x = alpha1*x + alpha2*y
// Replaces algebra.hpp for_each3 + shared_operations.hpp scale_sum2
// regions[0]: x (READ_WRITE)
// regions[1]: y (READ_ONLY)
// ============================================================
void axpy_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {

    AxpyArgs args;
    memcpy(&args, task->args, sizeof(AxpyArgs));

    const FieldAccessor<READ_WRITE, double, 1> x_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  y_acc(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        x_acc[*pir] = args.alpha1 * x_acc[*pir] + args.alpha2 * y_acc[*pir];
    }
}

// ============================================================
// Energy computation (inline, from system.hpp)
// ============================================================
double compute_energy(Runtime *runtime, Context ctx,
                      LogicalRegion q_lr, LogicalRegion p_lr, int N) {
    RegionRequirement q_req(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
    q_req.add_field(FID_VAL);
    RegionRequirement p_req(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
    p_req.add_field(FID_VAL);

    InlineLauncher q_launcher(q_req);
    InlineLauncher p_launcher(p_req);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_launcher);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_launcher);
    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(p_pr, FID_VAL);

    double energy = 0.5 * checked_math::pow(std::abs((double)q[0]), LAMBDA) / LAMBDA;
    for (int i = 0; i < N - 1; ++i) {
        double qi  = q[i];
        double qi1 = q[i + 1];
        double pi  = p[i];
        energy += 0.5 * pi * pi
            + checked_math::pow(qi, KAPPA) / KAPPA
            + checked_math::pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    double pN1 = p[N - 1];
    double qN1 = q[N - 1];
    energy += 0.5 * pN1 * pN1
        + checked_math::pow(qN1, KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(qN1), LAMBDA) / LAMBDA;

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    return energy;
}

// ============================================================
// Top-level task
// ============================================================
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {

    // Default parameters (matching HPX version)
    int N = 1024;
    int G = 128;
    int steps = 100;
    double dt = 0.01;

    // Parse command-line arguments
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--N") == 0 && i + 1 < command_args.argc) {
            N = atoi(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--G") == 0 && i + 1 < command_args.argc) {
            G = atoi(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--steps") == 0 && i + 1 < command_args.argc) {
            steps = atoi(command_args.argv[++i]);
        } else if (strcmp(command_args.argv[i], "--dt") == 0 && i + 1 < command_args.argc) {
            dt = atof(command_args.argv[++i]);
        }
    }

    int M = N / G;
    assert(M * G == N && "N must be divisible by G");

    // Open output file
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }
    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt << std::endl;

    // --------------------------------------------------------
    // Create index space [0, N-1] and field space
    // --------------------------------------------------------
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Three logical regions: q (position), p (momentum), dpdt (force)
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // --------------------------------------------------------
    // Disjoint partition into M equal blocks using restriction
    // --------------------------------------------------------
    IndexSpace color_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));

    Transform<1,1> block_tf;
    block_tf[0][0] = G;
    IndexPartition disjoint_ip = runtime->create_partition_by_restriction(
        ctx, is, color_is, block_tf, Rect<1>(0, G - 1));

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, disjoint_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, disjoint_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, disjoint_ip);

    // --------------------------------------------------------
    // Aliased ghost partition for q (extends 1 element into neighbors)
    // For color i, subspace = intersect([0,N-1], [i*G-1, i*G+G])
    // --------------------------------------------------------
    IndexPartition ghost_ip = runtime->create_partition_by_restriction(
        ctx, is, color_is, block_tf, Rect<1>(-1, G));
    LogicalPartition q_ghost_lp = runtime->get_logical_partition(ctx, q_lr, ghost_ip);

    // --------------------------------------------------------
    // Initialization
    // --------------------------------------------------------
    // q = 0
    runtime->fill_field<double>(ctx, q_lr, q_lr, FID_VAL, 0.0);

    // dpdt = 0
    runtime->fill_field<double>(ctx, dpdt_lr, dpdt_lr, FID_VAL, 0.0);

    // p = random (same seed and distribution as HPX version)
    {
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));

        RegionRequirement req(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr);
        req.add_field(FID_VAL);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, launcher);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> p_acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) {
            p_acc[i] = p_init[i];
        }
        runtime->unmap_region(ctx, pr);
    }

    // --------------------------------------------------------
    // Compute initial energy
    // --------------------------------------------------------
    double initial_energy = compute_energy(runtime, ctx, q_lr, p_lr, N);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(initial_energy)) << std::endl;

    // --------------------------------------------------------
    // Time integration: symplectic_rkn_sb3a_mclachlan stepper
    //
    // For each step, the stepper performs NUM_STAGES stages:
    //   for l = 0 .. NUM_STAGES-1:
    //     if l > 0: q = q + a[l-1]*dt * p
    //     dpdt = f(q)              (force computation)
    //     if b[l] != 0: p = p + b[l]*dt * dpdt
    //   q = q + a[NUM_STAGES-1]*dt * p   (final position update)
    //
    // Legion enforces ordering via region dependencies.
    // --------------------------------------------------------
    for (int step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {

            // --- Q update: q = 1.0*q + a[l-1]*dt * p ---
            if (l > 0) {
                AxpyArgs axpy_args;
                axpy_args.alpha1 = 1.0;
                axpy_args.alpha2 = coef_a[l - 1] * dt;

                IndexLauncher launcher(AXPY_TASK_ID, color_is,
                    TaskArgument(&axpy_args, sizeof(AxpyArgs)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_lp, 0 /*proj*/, READ_WRITE, EXCLUSIVE, q_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0 /*proj*/, READ_ONLY, EXCLUSIVE, p_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // --- Force computation: dpdt = f(q) ---
            {
                IndexLauncher launcher(FORCE_TASK_ID, color_is,
                    TaskArgument(NULL, 0), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_ghost_lp, 0 /*proj*/, READ_ONLY, SIMULTANEOUS, q_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0 /*proj*/, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // --- P update: p = 1.0*p + b[l]*dt * dpdt ---
            if (coef_b[l] != 0.0) {
                AxpyArgs axpy_args;
                axpy_args.alpha1 = 1.0;
                axpy_args.alpha2 = coef_b[l] * dt;

                IndexLauncher launcher(AXPY_TASK_ID, color_is,
                    TaskArgument(&axpy_args, sizeof(AxpyArgs)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0 /*proj*/, READ_WRITE, EXCLUSIVE, p_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0 /*proj*/, READ_ONLY, EXCLUSIVE, dpdt_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }
        }

        // --- Final Q update: q = 1.0*q + a[NUM_STAGES-1]*dt * p ---
        {
            AxpyArgs axpy_args;
            axpy_args.alpha1 = 1.0;
            axpy_args.alpha2 = coef_a[NUM_STAGES - 1] * dt;

            IndexLauncher launcher(AXPY_TASK_ID, color_is,
                TaskArgument(&axpy_args, sizeof(AxpyArgs)), ArgumentMap());
            launcher.add_region_requirement(
                RegionRequirement(q_lp, 0 /*proj*/, READ_WRITE, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(p_lp, 0 /*proj*/, READ_ONLY, EXCLUSIVE, p_lr));
            launcher.region_requirements[1].add_field(FID_VAL);
            runtime->execute_index_space(ctx, launcher);
        }
    }

    // --------------------------------------------------------
    // Compute final energy
    // --------------------------------------------------------
    double final_energy = compute_energy(runtime, ctx, q_lr, p_lr, N);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(final_energy)) << std::endl;

    outfile.close();

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ============================================================
// Main: register tasks and start Legion runtime
// ============================================================
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
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<force_task>(registrar, "force");
    }

    {
        TaskVariantRegistrar registrar(AXPY_TASK_ID, "axpy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<axpy_task>(registrar, "axpy");
    }

    return Runtime::start(argc, argv);
}
