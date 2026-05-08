// Copyright 2013 Mario Mulansky
// Translated from HPX execution model to Legion execution model
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <string>

#include <boost/math/special_functions/sign.hpp>

#include "legion.h"

using namespace Legion;

/* ================================================================
 * Constants and math helpers (from system.hpp)
 * ================================================================ */
static const double KAPPA  = 3.5;
static const double LAMBDA = 4.5;

namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k) {
    return checked_math::pow(x, k) * boost::math::sign(x);
}

/* ================================================================
 * Symplectic RKN SB3A McLachlan stepper coefficients
 * (from boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan)
 * ================================================================ */
static const int NUM_STAGES = 6;
static double COEF_A[NUM_STAGES];
static double COEF_B[NUM_STAGES];

static void init_stepper_coefficients() {
    COEF_A[0] =  0.40518861839525227722;
    COEF_A[1] = -0.28714404081652408900;
    COEF_A[2] =  1.0 - 2.0 * (COEF_A[0] + COEF_A[1]);
    COEF_A[3] =  COEF_A[2];
    COEF_A[4] =  COEF_A[1];
    COEF_A[5] =  COEF_A[0];

    COEF_B[0] = -3.0 / 73.0;
    COEF_B[1] =  17.0 / 59.0;
    COEF_B[2] =  1.0 - 2.0 * (COEF_B[0] + COEF_B[1]);
    COEF_B[3] =  COEF_B[1];
    COEF_B[4] =  COEF_B[0];
    COEF_B[5] =  0.0;
}

/* ================================================================
 * IDs
 * ================================================================ */
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
    SYSTEM_TASK_ID,
    SCALE_SUM2_TASK_ID,
    FMA_TASK_ID,
};

enum FieldIDs {
    FID_VAL,
};

/* ================================================================
 * Task argument PODs
 * ================================================================ */
struct SystemArgs {
    int block_idx;
    int num_blocks;
    int block_size;
    int total_N;
};

struct ScaleSum2Args {
    double alpha1;
    double alpha2;
};

struct FMAArgs {
    double alpha;
};

/* ================================================================
 * system_task  – compute force dpdt = f(q) for one block
 *
 * Mirrors osc_chain / system_first_block / system_center_block /
 * system_last_block from system.hpp.
 *
 * Region 0 : full q region   (READ_ONLY)   – allows ghost reads
 * Region 1 : dpdt sub-region (WRITE_DISCARD)
 * ================================================================ */
void system_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const SystemArgs a = *reinterpret_cast<const SystemArgs*>(task->args);
    const int idx = a.block_idx;
    const int M   = a.num_blocks;
    const int G   = a.block_size;
    const int N   = a.total_N;

    const FieldAccessor<READ_ONLY,     double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_VAL);

    const int start   = idx * G;
    const int local_N = (idx == M - 1) ? (N - start) : G;

    /* Left/right boundary values (ghost reads for coupling) */
    double q_l = (idx == 0)     ? 0.0 : (double)q[start - 1];
    double q_r = (idx == M - 1) ? 0.0 : (double)q[start + local_N];

    /* Accumulate coupling left→right across the block */
    double coupling = -signed_pow((double)q[start] - q_l, LAMBDA - 1);

    for (int i = 0; i < local_N - 1; ++i) {
        int gi = start + i;
        double val = -signed_pow((double)q[gi], KAPPA - 1) + coupling;
        coupling = signed_pow((double)q[gi] - (double)q[gi + 1], LAMBDA - 1);
        val -= coupling;
        dpdt[gi] = val;
    }

    /* Last element in this block */
    {
        int gi = start + local_N - 1;
        if (idx == M - 1)
            dpdt[gi] = -signed_pow((double)q[gi], KAPPA - 1)
                       + coupling
                       - signed_pow((double)q[gi], LAMBDA - 1);
        else
            dpdt[gi] = -signed_pow((double)q[gi], KAPPA - 1)
                       + coupling
                       - signed_pow((double)q[gi] - q_r, LAMBDA - 1);
    }
}

/* ================================================================
 * scale_sum2_task – dst = α₁·src1 + α₂·src2
 * Region 0 : dst  (WRITE_DISCARD)
 * Region 1 : src1 (READ_ONLY)
 * Region 2 : src2 (READ_ONLY)
 * ================================================================ */
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const ScaleSum2Args a = *reinterpret_cast<const ScaleSum2Args*>(task->args);
    const FieldAccessor<WRITE_DISCARD, double, 1> dst (regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,     double, 1> src1(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY,     double, 1> src2(regions[2], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++)
        dst[*it] = a.alpha1 * (double)src1[*it]
                 + a.alpha2 * (double)src2[*it];
}

/* ================================================================
 * fma_task – dst += α·src   (in-place update)
 * Region 0 : dst (READ_WRITE)
 * Region 1 : src (READ_ONLY)
 * ================================================================ */
void fma_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, Runtime *runtime)
{
    const FMAArgs a = *reinterpret_cast<const FMAArgs*>(task->args);
    const FieldAccessor<READ_WRITE, double, 1> dst(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> src(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++)
        dst[*it] = (double)dst[*it] + a.alpha * (double)src[*it];
}

/* ================================================================
 * Launch helpers (called from top-level task)
 * ================================================================ */

/* launch_system – M individual tasks, each reading the full q region
 * and writing to its own dpdt sub-region.  READ_ONLY on the same
 * parent region allows all M tasks to execute concurrently.           */
static void launch_system(Runtime *rt, Context ctx,
                          LogicalRegion q_lr,
                          LogicalRegion dpdt_lr,
                          LogicalPartition dpdt_lp,
                          int M, int G, int N)
{
    for (int i = 0; i < M; i++) {
        SystemArgs args;
        args.block_idx  = i;
        args.num_blocks = M;
        args.block_size = G;
        args.total_N    = N;

        LogicalRegion dpdt_sub =
            rt->get_logical_subregion_by_color(ctx, dpdt_lp, i);

        TaskLauncher launcher(SYSTEM_TASK_ID,
                              TaskArgument(&args, sizeof(args)));
        launcher.add_region_requirement(
            RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements[0].add_field(FID_VAL);
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements[1].add_field(FID_VAL);

        rt->execute_task(ctx, launcher);
    }
}

/* launch_scale_sum2 – index launch: dst = α₁·s1 + α₂·s2             */
static void launch_scale_sum2(Runtime *rt, Context ctx,
                               LogicalPartition dst_lp, LogicalRegion dst_lr,
                               LogicalPartition s1_lp,  LogicalRegion s1_lr,
                               LogicalPartition s2_lp,  LogicalRegion s2_lr,
                               IndexSpace color_is,
                               double a1, double a2)
{
    ScaleSum2Args args;
    args.alpha1 = a1;
    args.alpha2 = a2;

    IndexLauncher launcher(SCALE_SUM2_TASK_ID, color_is,
                           TaskArgument(&args, sizeof(args)),
                           ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(dst_lp, 0, WRITE_DISCARD, EXCLUSIVE, dst_lr));
    launcher.region_requirements[0].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(s1_lp, 0, READ_ONLY, EXCLUSIVE, s1_lr));
    launcher.region_requirements[1].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(s2_lp, 0, READ_ONLY, EXCLUSIVE, s2_lr));
    launcher.region_requirements[2].add_field(FID_VAL);

    rt->execute_index_space(ctx, launcher);
}

/* launch_fma – index launch: dst += α·src                             */
static void launch_fma(Runtime *rt, Context ctx,
                        LogicalPartition dst_lp, LogicalRegion dst_lr,
                        LogicalPartition src_lp, LogicalRegion src_lr,
                        IndexSpace color_is, double alpha)
{
    FMAArgs args;
    args.alpha = alpha;

    IndexLauncher launcher(FMA_TASK_ID, color_is,
                           TaskArgument(&args, sizeof(args)),
                           ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(dst_lp, 0, READ_WRITE, EXCLUSIVE, dst_lr));
    launcher.region_requirements[0].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(src_lp, 0, READ_ONLY, EXCLUSIVE, src_lr));
    launcher.region_requirements[1].add_field(FID_VAL);

    rt->execute_index_space(ctx, launcher);
}

/* ================================================================
 * energy – computed via inline mapping (matches original energy())
 * ================================================================ */
static double compute_energy(Runtime *rt, Context ctx,
                             LogicalRegion q_lr, LogicalRegion p_lr, int N)
{
    InlineLauncher ql(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    ql.requirement.add_field(FID_VAL);
    PhysicalRegion q_pr = rt->map_region(ctx, ql);
    q_pr.wait_until_valid();

    InlineLauncher pl(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    pl.requirement.add_field(FID_VAL);
    PhysicalRegion p_pr = rt->map_region(ctx, pl);
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> q(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(p_pr, FID_VAL);

    using checked_math::pow;

    double E = 0.5 * pow(std::abs((double)q[0]), LAMBDA) / LAMBDA;
    for (int i = 0; i < N - 1; ++i) {
        double qi  = q[i];
        double qi1 = q[i + 1];
        double pi  = p[i];
        E += 0.5 * pi * pi
           + pow(qi, KAPPA) / KAPPA
           + pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    {
        double qn = q[N - 1];
        double pn = p[N - 1];
        E += 0.5 * pn * pn
           + pow(qn, KAPPA) / KAPPA
           + 0.5 * pow(std::abs(qn), LAMBDA) / LAMBDA;
    }

    rt->unmap_region(ctx, q_pr);
    rt->unmap_region(ctx, p_pr);
    return E;
}

/* ================================================================
 * Top-level task
 *
 * Orchestrates initialisation, the symplectic_rkn_sb3a_mclachlan
 * integration loop, and energy output – mirroring hpx_main().
 *
 * The stepper performs 6 stages per step (position-force-momentum order):
 *   stage 0  :  qt = q + a0·dt·p;   dpdt=f(qt);  p += b0·dt·dpdt
 *   stage 1-4:  qt += al·dt·p;      dpdt=f(qt);  p += bl·dt·dpdt
 *   stage 5  :  q  = qt + a5·dt·p   (b5=0, no momentum update)
 * ================================================================ */
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    init_stepper_coefficients();

    /* ---------- command-line parsing ---------- */
    int    N     = 1024;
    int    G     = 128;
    int    steps = 100;
    double dt    = 0.01;

    const InputArgs &cargs = Runtime::get_input_args();
    for (int i = 1; i < cargs.argc; i++) {
        if (strcmp(cargs.argv[i], "--N") == 0 && i + 1 < cargs.argc)
            N = atoi(cargs.argv[++i]);
        else if (strncmp(cargs.argv[i], "--N=", 4) == 0)
            N = atoi(cargs.argv[i] + 4);
        else if (strcmp(cargs.argv[i], "--G") == 0 && i + 1 < cargs.argc)
            G = atoi(cargs.argv[++i]);
        else if (strncmp(cargs.argv[i], "--G=", 4) == 0)
            G = atoi(cargs.argv[i] + 4);
        else if (strcmp(cargs.argv[i], "--steps") == 0 && i + 1 < cargs.argc)
            steps = atoi(cargs.argv[++i]);
        else if (strncmp(cargs.argv[i], "--steps=", 8) == 0)
            steps = atoi(cargs.argv[i] + 8);
        else if (strcmp(cargs.argv[i], "--dt") == 0 && i + 1 < cargs.argc)
            dt = atof(cargs.argv[++i]);
        else if (strncmp(cargs.argv[i], "--dt=", 5) == 0)
            dt = atof(cargs.argv[i] + 5);
    }

    const int M = N / G;

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

    /* ---------- create regions ---------- */
    IndexSpace is     = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
    IndexSpace clr_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion qt_lr   = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    IndexPartition ip = runtime->create_equal_partition(ctx, is, clr_is);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    ip);
    LogicalPartition qt_lp   = runtime->get_logical_partition(ctx, qt_lr,   ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    /* ---------- initialise q = 0, p = random ---------- */
    {
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::mt19937 rng(0);
        auto gen = std::bind(dist, rng);
        std::generate(p_init.begin(), p_init.end(), std::ref(gen));

        InlineLauncher ql(RegionRequirement(q_lr, WRITE_DISCARD,
                                            EXCLUSIVE, q_lr));
        ql.requirement.add_field(FID_VAL);
        PhysicalRegion q_pr = runtime->map_region(ctx, ql);
        q_pr.wait_until_valid();

        InlineLauncher pl(RegionRequirement(p_lr, WRITE_DISCARD,
                                            EXCLUSIVE, p_lr));
        pl.requirement.add_field(FID_VAL);
        PhysicalRegion p_pr = runtime->map_region(ctx, pl);
        p_pr.wait_until_valid();

        {
            const FieldAccessor<WRITE_DISCARD, double, 1> qa(q_pr, FID_VAL);
            const FieldAccessor<WRITE_DISCARD, double, 1> pa(p_pr, FID_VAL);
            for (int i = 0; i < N; i++) {
                qa[i] = 0.0;
                pa[i] = p_init[i];
            }
        }

        runtime->unmap_region(ctx, q_pr);
        runtime->unmap_region(ctx, p_pr);
    }

    /* ---------- initial energy ---------- */
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(
                   compute_energy(runtime, ctx, q_lr, p_lr, N)))
            << std::endl;

    /* ==========================================================
     * Integration loop  –  symplectic_rkn_sb3a_mclachlan stepper
     *
     * Each do_step performs NUM_STAGES (6) stages
     * (position-force-momentum ordering, matching boost):
     *
     *   stage 0          : qt   = 1·q  + a[0]·dt · p
     *                      dpdt = f(qt)
     *                      p   += b[0]·dt · dpdt
     *
     *   stages 1 .. 4    : qt  += a[l]·dt · p
     *                      dpdt = f(qt)
     *                      p   += b[l]·dt · dpdt
     *
     *   stage 5 (final)  : q    = 1·qt + a[5]·dt · p
     *                      (b[5]=0, no momentum update needed)
     * ========================================================== */
    for (int step = 0; step < steps; ++step) {

        /* ---- stage 0 ---- */
        // qt = q + a[0]*dt*p
        launch_scale_sum2(runtime, ctx,
                          qt_lp, qt_lr,   q_lp, q_lr,   p_lp, p_lr,
                          clr_is, 1.0, dt * COEF_A[0]);
        // dpdt = f(qt)
        launch_system(runtime, ctx, qt_lr, dpdt_lr, dpdt_lp, M, G, N);
        // p += b[0]*dt*dpdt
        launch_fma(runtime, ctx,
                   p_lp, p_lr,   dpdt_lp, dpdt_lr,
                   clr_is, dt * COEF_B[0]);

        /* ---- stages 1 .. NUM_STAGES-2 ---- */
        for (int l = 1; l < NUM_STAGES - 1; ++l) {
            // qt += a[l]*dt*p
            launch_fma(runtime, ctx,
                       qt_lp, qt_lr,   p_lp, p_lr,
                       clr_is, dt * COEF_A[l]);
            // dpdt = f(qt)
            launch_system(runtime, ctx, qt_lr, dpdt_lr, dpdt_lp, M, G, N);
            // p += b[l]*dt*dpdt
            launch_fma(runtime, ctx,
                       p_lp, p_lr,   dpdt_lp, dpdt_lr,
                       clr_is, dt * COEF_B[l]);
        }

        /* ---- final stage (l = NUM_STAGES-1 = 5) ---- */
        // q = qt + a[5]*dt*p   (b[5]=0, no momentum update)
        launch_scale_sum2(runtime, ctx,
                          q_lp, q_lr,   qt_lp, qt_lr,   p_lp, p_lr,
                          clr_is, 1.0, dt * COEF_A[NUM_STAGES - 1]);
    }

    /* ---------- final energy ---------- */
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(
                   compute_energy(runtime, ctx, q_lr, p_lr, N)))
            << std::endl;

    outfile.close();

    /* ---------- cleanup ---------- */
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, qt_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, clr_is);
}

/* ================================================================
 * main – register tasks and start the Legion runtime
 * ================================================================ */
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar r(TOP_LEVEL_TASK_ID, "top_level_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(r, "top_level_task");
    }
    {
        TaskVariantRegistrar r(SYSTEM_TASK_ID, "system_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<system_task>(r, "system_task");
    }
    {
        TaskVariantRegistrar r(SCALE_SUM2_TASK_ID, "scale_sum2_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<scale_sum2_task>(r, "scale_sum2_task");
    }
    {
        TaskVariantRegistrar r(FMA_TASK_ID, "fma_task");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<fma_task>(r, "fma_task");
    }

    return Runtime::start(argc, argv);
}
