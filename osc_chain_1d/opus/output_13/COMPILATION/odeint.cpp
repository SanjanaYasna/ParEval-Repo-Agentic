// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <algorithm>
#include <cassert>

#include <legion.h>

using namespace Legion;

// Physics constants (matching system.hpp)
const double KAPPA = 3.5;
const double LAMBDA = 4.5;

enum {
    TOP_LEVEL_TASK_ID,
    FORCE_TASK_ID,
    SCALE_SUM2_TASK_ID,
};

enum {
    FID_VAL,
};

struct ForceArgs {
    int num_blocks;
    int block_size;
};

struct ScaleArgs {
    double alpha1;
    double alpha2;
};

// ---------- Math helpers (from system.hpp) ----------

namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    if (x > 0.0)
        return checked_math::pow(x, k);
    else if (x < 0.0)
        return -checked_math::pow(x, k);
    return 0.0;
}

// ---------- Energy computation (from system.hpp) ----------

double compute_energy(const double *q, const double *p, int N) {
    using checked_math::pow;
    double energy = 0.5 * pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (int i = 0; i < N - 1; ++i) {
        energy += 0.5 * p[i] * p[i]
                + pow(q[i], KAPPA) / KAPPA
                + pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    energy += 0.5 * p[N - 1] * p[N - 1]
            + pow(q[N - 1], KAPPA) / KAPPA
            + 0.5 * pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return energy;
}

// ---------- Force task (parallel over blocks, from system.hpp) ----------
// regions[0]: q with ghost cells (READ_ONLY, from aliased ghost partition)
// regions[1]: dpdt block (WRITE_DISCARD, from disjoint partition)

void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
    ForceArgs args = *(const ForceArgs *)task->args;
    int block_idx = task->index_point[0];
    int M = args.num_blocks;
    int G = args.block_size;

    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

    // Determine own and ghost element ranges
    Domain dpdt_dom = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> dpdt_rect = dpdt_dom;
    coord_t own_lo = dpdt_rect.lo[0];

    Domain q_dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> q_rect = q_dom;

    bool has_left  = (q_rect.lo[0] < dpdt_rect.lo[0]);
    bool has_right = (q_rect.hi[0] > dpdt_rect.hi[0]);

    // Read own q values into a local buffer
    std::vector<double> q(G);
    for (int i = 0; i < G; i++)
        q[i] = q_acc[own_lo + i];

    double q_left  = has_left  ? (double)q_acc[q_rect.lo[0]] : 0.0;
    double q_right = has_right ? (double)q_acc[q_rect.hi[0]] : 0.0;

    // Compute dpdt locally
    std::vector<double> dpdt(G);

    if (block_idx == 0) {
        // First block (system_first_block)
        double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
            dpdt[i] -= coupling_lr;
        }
        dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr
                     - signed_pow(q[G - 1] - q_right, LAMBDA - 1);
    } else if (block_idx == M - 1) {
        // Last block (system_last_block)
        double coupling_lr = -signed_pow(q[0] - q_left, LAMBDA - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
            dpdt[i] -= coupling_lr;
        }
        dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr
                     - signed_pow(q[G - 1], LAMBDA - 1);
    } else {
        // Center block (system_center_block)
        double coupling_lr = -signed_pow(q[0] - q_left, LAMBDA - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
            dpdt[i] -= coupling_lr;
        }
        dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr
                     - signed_pow(q[G - 1] - q_right, LAMBDA - 1);
    }

    // Write back
    for (int i = 0; i < G; i++)
        dpdt_acc[own_lo + i] = dpdt[i];
}

// ---------- Scale-sum2 task (from shared_operations.hpp) ----------
// s1 = alpha1 * s1 + alpha2 * s3
// regions[0]: s1 (READ_WRITE)
// regions[1]: s3 (READ_ONLY)

void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
    ScaleArgs args = *(const ScaleArgs *)task->args;
    double alpha1 = args.alpha1;
    double alpha2 = args.alpha2;

    const FieldAccessor<READ_WRITE, double, 1> s1_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>  s3_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;

    for (PointInRectIterator<1> pir(rect); pir(); pir++) {
        double v1 = s1_acc[*pir];
        double v3 = s3_acc[*pir];
        s1_acc[*pir] = alpha1 * v1 + alpha2 * v3;
    }
}

// ---------- Top-level task ----------

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
    // Parse command-line arguments (--N, --G, --steps, --dt)
    int N = 1024;
    int G = 128;
    int steps = 100;
    double dt = 0.01;

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

    // Output file
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        fprintf(stderr, "Failed to open odeint.txt for writing.\n");
        return;
    }
    outfile << "Dimension: " << N
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M
            << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    // ---- Create index space and field space ----
    Rect<1> elem_rect(0, N - 1);
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
    runtime->attach_name(is, "element_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // Logical regions for q, p, dpdt
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // ---- Create partitions ----
    Rect<1> color_rect(0, M - 1);
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);

    // Disjoint equal partition for q, p, dpdt
    IndexPartition disjoint_ip = runtime->create_equal_partition(ctx, is, color_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    disjoint_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    disjoint_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, disjoint_ip);

    // Aliased (overlapping) ghost partition for force computation reads of q.
    // Block i needs elements [max(0, i*G-1), min(N-1, (i+1)*G)].
    // Using create_partition_by_restriction: for color c, sub-region = transform*c + extent,
    // clipped to parent index space.
    // transform maps color i -> i*G, extent is [-1, G].
    // Result for color i: [i*G - 1, i*G + G] intersected with [0, N-1].
    Transform<1,1> ghost_transform;
    ghost_transform[0][0] = G;
    Rect<1> ghost_extent(Point<1>(-1), Point<1>(G));
    IndexPartition ghost_ip = runtime->create_partition_by_restriction(
        ctx, is, color_is, ghost_transform, ghost_extent);
    LogicalPartition q_ghost_lp = runtime->get_logical_partition(ctx, q_lr, ghost_ip);

    // ---- Initialize q (zeros) ----
    {
        RegionRequirement req(q_lr, WRITE_DISCARD, EXCLUSIVE, q_lr);
        req.add_field(FID_VAL);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, launcher);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = 0.0;
        runtime->unmap_region(ctx, pr);
    }

    // ---- Initialize p (random, matching HPX code's MT19937 seed=0) ----
    std::vector<double> p_init(N);
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));
    }
    {
        RegionRequirement req(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr);
        req.add_field(FID_VAL);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, launcher);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = p_init[i];
        runtime->unmap_region(ctx, pr);
    }

    // ---- Initialize dpdt (zeros) ----
    {
        RegionRequirement req(dpdt_lr, WRITE_DISCARD, EXCLUSIVE, dpdt_lr);
        req.add_field(FID_VAL);
        InlineLauncher launcher(req);
        PhysicalRegion pr = runtime->map_region(ctx, launcher);
        pr.wait_until_valid();
        const FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
        for (int i = 0; i < N; i++) acc[i] = 0.0;
        runtime->unmap_region(ctx, pr);
    }

    // ---- Compute and print initial energy ----
    {
        RegionRequirement req_q(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
        req_q.add_field(FID_VAL);
        RegionRequirement req_p(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
        req_p.add_field(FID_VAL);
        PhysicalRegion pr_q = runtime->map_region(ctx, InlineLauncher(req_q));
        PhysicalRegion pr_p = runtime->map_region(ctx, InlineLauncher(req_p));
        pr_q.wait_until_valid();
        pr_p.wait_until_valid();
        const FieldAccessor<READ_ONLY, double, 1> q_acc(pr_q, FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> p_acc(pr_p, FID_VAL);
        std::vector<double> q(N), p(N);
        for (int i = 0; i < N; i++) { q[i] = q_acc[i]; p[i] = p_acc[i]; }
        double e = compute_energy(q.data(), p.data(), N);
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
        runtime->unmap_region(ctx, pr_q);
        runtime->unmap_region(ctx, pr_p);
    }

    // ---- Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages) ----
    // Matches boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan
    const int num_stages = 6;
    const double a_coef[6] = {
         0.40518861839525227722,
        -0.28714404081652408900,
         0.5 - (0.40518861839525227722 - 0.28714404081652408900),
         0.5 - (0.40518861839525227722 - 0.28714404081652408900),
        -0.28714404081652408900,
         0.40518861839525227722
    };
    const double b_coef[6] = {
        -3.0 / 73.0,
         17.0 / 59.0,
         1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
         17.0 / 59.0,
        -3.0 / 73.0,
         0.0
    };

    // The odeint symplectic_nystroem_stepper_base::do_step does:
    //   for l = 0 .. num_stages-1:
    //     if l > 0: q = q + a[l]*dt * p
    //     dpdt = f(q)
    //     p = p + b[l]*dt * dpdt
    //   q = q + a[0]*dt * p

    // ---- Time integration loop (integrate_n_steps) ----
    for (int step = 0; step < steps; step++) {
        for (int l = 0; l < num_stages; l++) {

            // Position update: q = 1.0*q + a[l]*dt * p  (skip for l == 0)
            if (l > 0) {
                ScaleArgs sargs;
                sargs.alpha1 = 1.0;
                sargs.alpha2 = a_coef[l] * dt;
                IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                    Domain(color_rect),
                    TaskArgument(&sargs, sizeof(sargs)),
                    ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_lp, 0/*identity proj*/, READ_WRITE, EXCLUSIVE, q_lr));
                launcher.region_requirements[0].add_field(FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_ONLY, EXCLUSIVE, p_lr));
                launcher.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // Force computation + momentum update (skip entirely if b[l] == 0)
            if (b_coef[l] != 0.0) {
                // f(q) -> dpdt
                {
                    ForceArgs fargs;
                    fargs.num_blocks = M;
                    fargs.block_size = G;
                    IndexLauncher launcher(FORCE_TASK_ID,
                        Domain(color_rect),
                        TaskArgument(&fargs, sizeof(fargs)),
                        ArgumentMap());
                    // q with ghosts (READ_ONLY, aliased partition — must use SIMULTANEOUS)
                    launcher.add_region_requirement(
                        RegionRequirement(q_ghost_lp, 0, READ_ONLY, SIMULTANEOUS, q_lr));
                    launcher.region_requirements[0].add_field(FID_VAL);
                    // dpdt (WRITE_DISCARD, disjoint partition)
                    launcher.add_region_requirement(
                        RegionRequirement(dpdt_lp, 0, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
                    launcher.region_requirements[1].add_field(FID_VAL);
                    runtime->execute_index_space(ctx, launcher);
                }
                // p = 1.0*p + b[l]*dt * dpdt
                {
                    ScaleArgs sargs;
                    sargs.alpha1 = 1.0;
                    sargs.alpha2 = b_coef[l] * dt;
                    IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                        Domain(color_rect),
                        TaskArgument(&sargs, sizeof(sargs)),
                        ArgumentMap());
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

        // Final position update: q = q + a[0]*dt * p
        {
            ScaleArgs sargs;
            sargs.alpha1 = 1.0;
            sargs.alpha2 = a_coef[0] * dt;
            IndexLauncher launcher(SCALE_SUM2_TASK_ID,
                Domain(color_rect),
                TaskArgument(&sargs, sizeof(sargs)),
                ArgumentMap());
            launcher.add_region_requirement(
                RegionRequirement(q_lp, 0, READ_WRITE, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            launcher.add_region_requirement(
                RegionRequirement(p_lp, 0, READ_ONLY, EXCLUSIVE, p_lr));
            launcher.region_requirements[1].add_field(FID_VAL);
            runtime->execute_index_space(ctx, launcher);
        }
    }

    // ---- Compute and print final energy ----
    {
        RegionRequirement req_q(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
        req_q.add_field(FID_VAL);
        RegionRequirement req_p(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
        req_p.add_field(FID_VAL);
        PhysicalRegion pr_q = runtime->map_region(ctx, InlineLauncher(req_q));
        PhysicalRegion pr_p = runtime->map_region(ctx, InlineLauncher(req_p));
        pr_q.wait_until_valid();
        pr_p.wait_until_valid();
        const FieldAccessor<READ_ONLY, double, 1> q_acc(pr_q, FID_VAL);
        const FieldAccessor<READ_ONLY, double, 1> p_acc(pr_p, FID_VAL);
        std::vector<double> q(N), p(N);
        for (int i = 0; i < N; i++) { q[i] = q_acc[i]; p[i] = p_acc[i]; }
        double e = compute_energy(q.data(), p.data(), N);
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
        runtime->unmap_region(ctx, pr_q);
        runtime->unmap_region(ctx, pr_p);
    }

    outfile.close();

    // ---- Cleanup ----
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ---------- Main: register tasks and start Legion runtime ----------

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
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }

    return Runtime::start(argc, argv);
}
