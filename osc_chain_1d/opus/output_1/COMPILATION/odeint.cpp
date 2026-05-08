// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <legion.h>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <cmath>
#include <cstring>
#include <functional>
#include <algorithm>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ===== Physics constants =====
static const double KAPPA = 3.5;
static const double LAMBDA_EXP = 4.5;

// ===== Stepper coefficients for symplectic_rkn_sb3a_mclachlan =====
static const double COEFF_A1 = 0.40518861839525227722;
static const double COEFF_A2 = -0.28714404081652408900;
static const double COEFF_A3 = 0.5 - (COEFF_A1 + COEFF_A2);
static const double COEFF_A[6] = {COEFF_A1, COEFF_A2, COEFF_A3,
                                   COEFF_A3, COEFF_A2, COEFF_A1};

static const double COEFF_B1 = -3.0 / 73.0;
static const double COEFF_B2 = 17.0 / 59.0;
static const double COEFF_B3 = 1.0 - 2.0 * (COEFF_B1 + COEFF_B2);
static const double COEFF_B[6] = {COEFF_B1, COEFF_B2, COEFF_B3,
                                   COEFF_B2, COEFF_B1, 0.0};

// ===== IDs =====
enum FieldIDs { FID_VAL = 100 };
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    INIT_ZERO_TASK_ID,
    INIT_DATA_TASK_ID,
    FORCE_TASK_ID,
    SCALE_SUM2_TASK_ID,
    ENERGY_TASK_ID,
};

// ===== Math helpers =====
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

// ===== Task argument structs =====
struct ForceArgs {
    size_t N, G, M, block_idx;
};

struct ScaleSum2Args {
    double alpha1, alpha2;
};

// ===== Leaf task implementations =====

void init_zero_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc[*pir] = 0.0;
}

void init_data_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const double *data = (const double *)task->args;
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc[*pir] = data[(*pir)[0]];
}

void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceArgs args = *(const ForceArgs *)task->args;
    size_t block_idx = args.block_idx;
    size_t N = args.N, G = args.G, M = args.M;

    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_VAL);

    coord_t start = (coord_t)(block_idx * G);
    coord_t end   = start + (coord_t)G;

    // Left boundary: coupling with neighbour (0 = fixed wall for first block)
    double q_left = (block_idx > 0) ? (double)q[start - 1] : 0.0;
    double coupling = -signed_pow((double)q[start] - q_left, LAMBDA_EXP - 1);

    for (coord_t i = start; i < end - 1; ++i) {
        double val = -signed_pow((double)q[i], KAPPA - 1) + coupling;
        coupling = signed_pow((double)q[i] - (double)q[i + 1], LAMBDA_EXP - 1);
        val -= coupling;
        dpdt[i] = val;
    }

    // Right boundary: coupling with neighbour (0 = fixed wall for last block)
    double q_right = (block_idx < M - 1) ? (double)q[end] : 0.0;
    dpdt[end - 1] = -signed_pow((double)q[end - 1], KAPPA - 1)
        + coupling - signed_pow((double)q[end - 1] - q_right, LAMBDA_EXP - 1);
}

void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    ScaleSum2Args args = *(const ScaleSum2Args *)task->args;
    const FieldAccessor<READ_WRITE, double, 1> x1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> x3(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        x1[*pir] = args.alpha1 * (double)x1[*pir] + args.alpha2 * (double)x3[*pir];
}

double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    coord_t lo = rect.lo[0];
    coord_t hi = rect.hi[0];

    double e = 0.5 * checked_math::pow(std::abs((double)q[lo]), LAMBDA_EXP)
               / LAMBDA_EXP;
    for (coord_t i = lo; i < hi; ++i) {
        double qi = q[i], pi = p[i], qi1 = q[i + 1];
        e += 0.5 * pi * pi
            + checked_math::pow(qi, KAPPA) / KAPPA
            + checked_math::pow(std::abs(qi - qi1), LAMBDA_EXP) / LAMBDA_EXP;
    }
    double qhi = q[hi], phi_val = p[hi];
    e += 0.5 * phi_val * phi_val
        + checked_math::pow(qhi, KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(qhi), LAMBDA_EXP) / LAMBDA_EXP;

    return e;
}

// ===== Top-level task =====

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // --- Parse command-line arguments ---
    size_t N = 1024, G = 128, steps = 100;
    double dt = 0.01;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (!strcmp(command_args.argv[i], "--N") && i + 1 < command_args.argc)
            N = (size_t)atol(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--G") && i + 1 < command_args.argc)
            G = (size_t)atol(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--steps") && i + 1 < command_args.argc)
            steps = (size_t)atol(command_args.argv[++i]);
        else if (!strcmp(command_args.argv[i], "--dt") && i + 1 < command_args.argc)
            dt = atof(command_args.argv[++i]);
    }

    const size_t M = N / G;

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

    // --- Create index spaces and field spaces ---
    Rect<1> elem_rect(0, (coord_t)(N - 1));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    // --- Create logical regions: q (position), p (momentum), dpdt (force) ---
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // --- Create equal partition into M blocks of size G ---
    Rect<1> color_rect(0, (coord_t)(M - 1));
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);

    Transform<1, 1> transform;
    transform[0][0] = (coord_t)G;
    Rect<1> extent(0, (coord_t)(G - 1));
    IndexPartition ip = runtime->create_partition_by_restriction(
        ctx, is, color_is, transform, extent);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // --- Initialize q to zero (index launch) ---
    {
        IndexLauncher launcher(INIT_ZERO_TASK_ID, color_is,
            TaskArgument(NULL, 0), ArgumentMap());
        launcher.add_region_requirement(
            RegionRequirement(q_lp, 0, WRITE_DISCARD, EXCLUSIVE, q_lr));
        launcher.add_field(0, FID_VAL);
        runtime->execute_index_space(ctx, launcher);
    }

    // --- Initialize p with random data (single task) ---
    {
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));

        TaskLauncher launcher(INIT_DATA_TASK_ID,
            TaskArgument(p_init.data(), N * sizeof(double)));
        launcher.add_region_requirement(
            RegionRequirement(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr));
        launcher.add_field(0, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    // --- Compute initial energy ---
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.add_field(1, FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // ===== Time integration (symplectic_rkn_sb3a_mclachlan) =====
    //
    // Algorithm per time-step (DKD ordering matching Boost):
    //   for l = 0..5:
    //     q = q + a[l]*dt * p              (position drift)
    //     if b[l] != 0:
    //       dpdt = f(q)                    (force evaluation)
    //       p = p + b[l]*dt * dpdt         (momentum kick)

    for (size_t step = 0; step < steps; ++step) {
        for (size_t l = 0; l < 6; ++l) {

            // --- Position drift: q = q + a[l]*dt * p ---
            {
                ScaleSum2Args ss;
                ss.alpha1 = 1.0;
                ss.alpha2 = COEFF_A[l] * dt;
                IndexLauncher launcher(SCALE_SUM2_TASK_ID, color_is,
                    TaskArgument(&ss, sizeof(ss)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_lp, 0, READ_WRITE, EXCLUSIVE, q_lr));
                launcher.add_field(0, FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_ONLY, EXCLUSIVE, p_lr));
                launcher.add_field(1, FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // --- Force evaluation and momentum kick (skip when b[l] == 0) ---
            if (COEFF_B[l] != 0.0) {
                // Compute force: dpdt = f(q)
                // Individual task launches so each reads full q (READ_ONLY)
                for (size_t i = 0; i < M; i++) {
                    ForceArgs fa;
                    fa.N = N; fa.G = G; fa.M = M; fa.block_idx = i;
                    TaskLauncher launcher(FORCE_TASK_ID,
                        TaskArgument(&fa, sizeof(fa)));
                    launcher.add_region_requirement(
                        RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
                    launcher.add_field(0, FID_VAL);
                    LogicalRegion dpdt_sub =
                        runtime->get_logical_subregion_by_color(ctx, dpdt_lp,
                            DomainPoint(Point<1>((coord_t)i)));
                    launcher.add_region_requirement(
                        RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE,
                                          dpdt_lr));
                    launcher.add_field(1, FID_VAL);
                    runtime->execute_task(ctx, launcher);
                }

                // Update momentum: p = p + b[l]*dt * dpdt
                ScaleSum2Args ss;
                ss.alpha1 = 1.0;
                ss.alpha2 = COEFF_B[l] * dt;
                IndexLauncher launcher(SCALE_SUM2_TASK_ID, color_is,
                    TaskArgument(&ss, sizeof(ss)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_WRITE, EXCLUSIVE, p_lr));
                launcher.add_field(0, FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0, READ_ONLY, EXCLUSIVE,
                                      dpdt_lr));
                launcher.add_field(1, FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }
        }
    }

    // --- Compute final energy ---
    {
        TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
        launcher.add_field(1, FID_VAL);
        Future f = runtime->execute_task(ctx, launcher);
        double e = f.get_result<double>();
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
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

// ===== Main and task registration =====

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
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
