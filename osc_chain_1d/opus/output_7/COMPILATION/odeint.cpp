// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <random>
#include <functional>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <map>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ── Constants ────────────────────────────────────────────────────────────────
const double KAPPA  = 3.5;
const double LAMBDA = 4.5;
const int    NUM_STAGES = 6;

// ── Hardcoded coefficients for symplectic_rkn_sb3a_mclachlan ─────────────────
// From boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp
static const double RKN_A[6] = {
    0.40518861839525227722,
    -0.28714404081652408900,
    0.5 - (0.40518861839525227722 - 0.28714404081652408900),
    0.5 - (0.40518861839525227722 - 0.28714404081652408900),
    -0.28714404081652408900,
    0.40518861839525227722
};

static const double RKN_B[6] = {
    -3.0/73.0,
    17.0/59.0,
    1.0 - 2.0*(17.0/59.0 - 3.0/73.0),
    1.0 - 2.0*(17.0/59.0 - 3.0/73.0),
    17.0/59.0,
    -3.0/73.0
};

// ── IDs ──────────────────────────────────────────────────────────────────────
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    SCALE_SUM2_TASK_ID,
    SYSTEM_TASK_ID,
};

enum FieldIDs {
    FID_VAL = 0,
};

// ── Math helpers (same as original system.hpp) ───────────────────────────────
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

static double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

// ── Argument structs passed to tasks ─────────────────────────────────────────
struct ScaleArgs {
    double alpha1;
    double alpha2;
};

struct SystemArgs {
    size_t N;
    size_t G;
    size_t M;
};

// ── scale_sum2 task : x[i] = alpha1*x[i] + alpha2*y[i] ─────────────────────
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    ScaleArgs args;
    assert(task->arglen == sizeof(ScaleArgs));
    memcpy(&args, task->args, sizeof(ScaleArgs));

    const FieldAccessor<READ_WRITE,double,1> x(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double,1> y(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);

    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        x[*pir] = args.alpha1 * (double)x[*pir] + args.alpha2 * (double)y[*pir];
}

// ── system evaluation task (one per block) ───────────────────────────────────
//  Region 0 : q  (ghost / aliased partition – READ_ONLY)
//  Region 1 : dpdt (disjoint partition     – WRITE_DISCARD)
void system_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    SystemArgs args;
    assert(task->arglen == sizeof(SystemArgs));
    memcpy(&args, task->args, sizeof(SystemArgs));

    const size_t N = args.N;
    const size_t G = args.G;
    const size_t M = args.M;

    int block_idx = task->index_point[0];
    size_t lo = (size_t)block_idx * G;
    size_t hi = lo + G - 1;   // inclusive

    const FieldAccessor<READ_ONLY,  double,1> q_acc(regions[0],   FID_VAL);
    const FieldAccessor<WRITE_DISCARD,double,1> dpdt_acc(regions[1], FID_VAL);

    // ── copy own q block into local vector for fast access ──
    std::vector<double> q(G);
    for (size_t i = 0; i < G; i++) q[i] = q_acc[lo + i];

    // ── boundary values ──
    double q_left  = (block_idx > 0)         ? (double)q_acc[lo - 1] : 0.0;
    double q_right = (block_idx < (int)M - 1)? (double)q_acc[hi + 1] : 0.0;

    // ── compute dpdt (mirrors system_first / center / last _block) ──
    std::vector<double> dpdt(G);

    double coupling_lr;
    if (block_idx == 0)
        coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    else
        coupling_lr = -signed_pow(q[0] - q_left, LAMBDA - 1);

    for (size_t i = 0; i < G - 1; i++) {
        dpdt[i]  = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }

    if (block_idx == (int)M - 1)
        dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1)
                     + coupling_lr
                     - signed_pow(q[G - 1], LAMBDA - 1);
    else
        dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1)
                     + coupling_lr
                     - signed_pow(q[G - 1] - q_right, LAMBDA - 1);

    for (size_t i = 0; i < G; i++) dpdt_acc[lo + i] = dpdt[i];
}

// ── energy (serial, inline‑mapped in top‑level task) ─────────────────────────
static double compute_energy(Runtime *runtime, Context ctx,
                             LogicalRegion q_lr, LogicalRegion p_lr,
                             size_t N)
{
    InlineLauncher q_il(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    q_il.add_field(FID_VAL);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_il);

    InlineLauncher p_il(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    p_il.add_field(FID_VAL);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_il);

    q_pr.wait_until_valid();
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY,double,1> qa(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY,double,1> pa(p_pr, FID_VAL);

    using checked_math::pow;
    double E = 0.5 * pow(std::abs((double)qa[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; i++) {
        double qi  = qa[i];
        double qi1 = qa[i + 1];
        double pi  = pa[i];
        E += 0.5*pi*pi
           + pow(qi, KAPPA) / KAPPA
           + pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    {
        double qN1 = qa[N - 1];
        double pN1 = pa[N - 1];
        E += 0.5*pN1*pN1
           + pow(qN1, KAPPA) / KAPPA
           + 0.5*pow(std::abs(qN1), LAMBDA) / LAMBDA;
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);
    return E;
}

// ── top‑level task ───────────────────────────────────────────────────────────
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ── parse command line ──
    size_t N     = 1024;
    size_t G     = 128;
    size_t steps = 100;
    double dt    = 0.01;

    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
        if (strcmp(command_args.argv[i], "--N") == 0 && i+1 < command_args.argc)
            N = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--G") == 0 && i+1 < command_args.argc)
            G = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--steps") == 0 && i+1 < command_args.argc)
            steps = (size_t)atol(command_args.argv[++i]);
        else if (strcmp(command_args.argv[i], "--dt") == 0 && i+1 < command_args.argc)
            dt = atof(command_args.argv[++i]);
    }
    const size_t M = N / G;

    // ── open output file ──
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

    // ── create index space, field space, regions ──
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, (coord_t)(N - 1)));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // ── disjoint equal partition (M blocks of G) ──
    IndexSpace color_is = runtime->create_index_space(ctx,
                              Rect<1>(0, (coord_t)(M - 1)));
    IndexPartition disjoint_ip = runtime->create_equal_partition(ctx, is, color_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    disjoint_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    disjoint_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, disjoint_ip);

    // ── ghost (aliased) partition for q – each block includes neighbour
    //    boundary elements ──
    std::map<DomainPoint, Domain> ghost_coloring;
    for (size_t i = 0; i < M; i++) {
        coord_t lo = (i > 0)     ? (coord_t)(i * G - 1) : 0;
        coord_t hi = (i < M - 1) ? (coord_t)((i + 1) * G) : (coord_t)(N - 1);
        ghost_coloring[DomainPoint(Point<1>((coord_t)i))] = Domain(Rect<1>(lo, hi));
    }
    IndexPartition ghost_ip = runtime->create_partition_by_domain(
        ctx, is, ghost_coloring, color_is, true, LEGION_ALIASED_COMPLETE_KIND);
    LogicalPartition q_ghost_lp = runtime->get_logical_partition(ctx, q_lr, ghost_ip);

    // ── initialise q = 0, p = random (Mersenne‑twister seed 0, U[-1,1]) ──
    {
        // Generate p_init on the host exactly as the original
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));

        // inline‑map and fill
        InlineLauncher q_il(RegionRequirement(q_lr, WRITE_DISCARD, EXCLUSIVE, q_lr));
        q_il.add_field(FID_VAL);
        PhysicalRegion q_pr = runtime->map_region(ctx, q_il);

        InlineLauncher p_il(RegionRequirement(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr));
        p_il.add_field(FID_VAL);
        PhysicalRegion p_pr = runtime->map_region(ctx, p_il);

        q_pr.wait_until_valid();
        p_pr.wait_until_valid();

        {
            const FieldAccessor<WRITE_DISCARD,double,1> qa(q_pr, FID_VAL);
            const FieldAccessor<WRITE_DISCARD,double,1> pa(p_pr, FID_VAL);
            for (size_t i = 0; i < N; i++) {
                qa[i] = 0.0;
                pa[i] = p_init[i];
            }
        }
        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    // ── initial energy ──
    {
        double E = compute_energy(runtime, ctx, q_lr, p_lr, N);
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
    }

    // ── symplectic integration loop ──
    Domain color_domain = runtime->get_index_space_domain(ctx, color_is);
    SystemArgs sys_args;
    sys_args.N = N; sys_args.G = G; sys_args.M = M;

    for (size_t step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {

            // ── q = 1·q + a[l]·dt · p ──
            {
                double a_dt = RKN_A[l] * dt;
                ScaleArgs sa; sa.alpha1 = 1.0; sa.alpha2 = a_dt;
                IndexLauncher launcher(SCALE_SUM2_TASK_ID, color_domain,
                    TaskArgument(&sa, sizeof(sa)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_lp, 0, READ_WRITE, EXCLUSIVE, q_lr));
                launcher.add_field(0, FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_ONLY, EXCLUSIVE, p_lr));
                launcher.add_field(1, FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // ── dpdt = f(q) ──
            {
                IndexLauncher launcher(SYSTEM_TASK_ID, color_domain,
                    TaskArgument(&sys_args, sizeof(sys_args)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(q_ghost_lp, 0, READ_ONLY, EXCLUSIVE, q_lr));
                launcher.add_field(0, FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
                launcher.add_field(1, FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }

            // ── p = 1·p + b[l]·dt · dpdt ──
            {
                double b_dt = RKN_B[l] * dt;
                ScaleArgs sa; sa.alpha1 = 1.0; sa.alpha2 = b_dt;
                IndexLauncher launcher(SCALE_SUM2_TASK_ID, color_domain,
                    TaskArgument(&sa, sizeof(sa)), ArgumentMap());
                launcher.add_region_requirement(
                    RegionRequirement(p_lp, 0, READ_WRITE, EXCLUSIVE, p_lr));
                launcher.add_field(0, FID_VAL);
                launcher.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0, READ_ONLY, EXCLUSIVE, dpdt_lr));
                launcher.add_field(1, FID_VAL);
                runtime->execute_index_space(ctx, launcher);
            }
        }
    }

    // ── final energy ──
    {
        double E = compute_energy(runtime, ctx, q_lr, p_lr, N);
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
    }

    outfile.close();

    // ── clean up ──
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ── main & registration ──────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_TASK_ID, "system");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_task>(registrar, "system");
    }

    return Runtime::start(argc, argv);
}
