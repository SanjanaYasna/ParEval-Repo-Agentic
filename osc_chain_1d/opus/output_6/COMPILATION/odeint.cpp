// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <functional>
#include <algorithm>

#include "legion.h"
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

/* ================================================================
   Physics constants
   ================================================================ */
static const double KAPPA  = 3.5;
static const double LAMBDA = 4.5;

/* ================================================================
   Symplectic RKN SB3A McLachlan coefficients (6 stages)
   See boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp
   Algorithm per step:
     for s = 0 .. 5:
       q += a[s]*dt * p          (drift)
       dpdt = F(q)               (force)
       p += b[s+1]*dt * dpdt     (kick)
   Note: b[0] = 0 so no initial kick.
   ================================================================ */
static const int NUM_STAGES = 6;
static const double coef_a[6] = {
     0.40518861839525227722,
    -0.28714404081652408900,
     0.38196542232527382016,
     0.38196542232527382016,
    -0.28714404081652408900,
     0.40518861839525227722
};
static const double coef_b[7] = {
     0.0,
     0.74167036435061295345,
    -0.40910082580003159400,
     0.16743046145041864055,
     0.16743046145041864055,
    -0.40910082580003159400,
     0.74167036435061295345
};

/* ================================================================
   IDs
   ================================================================ */
enum { FID_VAL = 0 };
enum TaskIDs  { TID_TOP = 0, TID_FORCE, TID_DRIFT, TID_KICK, TID_ENERGY };

/* ================================================================
   Math helpers (same as original system.hpp)
   ================================================================ */
namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

static inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

/* ================================================================
   Task-argument structs
   ================================================================ */
struct ForceArgs     { int N; int G; int M; };
struct DriftKickArgs { double alpha; int G; };

/* ================================================================
   force_task – compute dpdt = F(q) for one block
   regions[0]: q  via ghost partition  (READ_ONLY)
   regions[1]: dpdt via disjoint part  (WRITE_DISCARD)
   ================================================================ */
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceArgs fa = *(const ForceArgs *)task->args;
    int idx = task->index_point[0];
    coord_t lo = (coord_t)idx * fa.G;
    coord_t hi = lo + fa.G - 1;

    const FieldAccessor<READ_ONLY,     double, 1> q (regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dp(regions[1], FID_VAL);

    /* left coupling */
    double clr;
    if (idx == 0)
        clr = -signed_pow(q[lo], LAMBDA - 1);
    else
        clr = -signed_pow(q[lo] - q[lo - 1], LAMBDA - 1);

    /* interior of block */
    for (coord_t j = lo; j < hi; ++j) {
        double val = -signed_pow(q[j], KAPPA - 1) + clr;
        clr        =  signed_pow(q[j] - q[j + 1], LAMBDA - 1);
        dp[j]      = val - clr;
    }

    /* right boundary */
    if (idx < fa.M - 1)
        dp[hi] = -signed_pow(q[hi], KAPPA - 1) + clr
                 - signed_pow(q[hi] - q[hi + 1], LAMBDA - 1);
    else
        dp[hi] = -signed_pow(q[hi], KAPPA - 1) + clr
                 - signed_pow(q[hi], LAMBDA - 1);
}

/* ================================================================
   drift_task – q += alpha * p   (one block)
   regions[0]: q  (READ_WRITE)
   regions[1]: p  (READ_ONLY)
   ================================================================ */
void drift_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    DriftKickArgs da = *(const DriftKickArgs *)task->args;
    int idx = task->index_point[0];
    coord_t lo = (coord_t)idx * da.G;
    coord_t hi = lo + da.G - 1;

    const FieldAccessor<READ_WRITE, double, 1> qf(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> pf(regions[1], FID_VAL);

    for (coord_t j = lo; j <= hi; ++j)
        qf[j] = (double)qf[j] + da.alpha * (double)pf[j];
}

/* ================================================================
   kick_task – p += alpha * dpdt   (one block)
   regions[0]: p    (READ_WRITE)
   regions[1]: dpdt (READ_ONLY)
   ================================================================ */
void kick_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
    DriftKickArgs da = *(const DriftKickArgs *)task->args;
    int idx = task->index_point[0];
    coord_t lo = (coord_t)idx * da.G;
    coord_t hi = lo + da.G - 1;

    const FieldAccessor<READ_WRITE, double, 1> pf(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> df(regions[1], FID_VAL);

    for (coord_t j = lo; j <= hi; ++j)
        pf[j] = (double)pf[j] + da.alpha * (double)df[j];
}

/* ================================================================
   energy_task – compute total energy (single task, whole region)
   regions[0]: q  (READ_ONLY)
   regions[1]: p  (READ_ONLY)
   Returns double.
   ================================================================ */
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    int N = *(const int *)task->args;
    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_VAL);

    using checked_math::pow;
    using std::abs;

    double E = 0.5 * pow(abs((double)q[0]), LAMBDA) / LAMBDA;
    for (int i = 0; i < N - 1; ++i) {
        double qi  = q[Point<1>(i)];
        double qi1 = q[Point<1>(i + 1)];
        double pi  = p[Point<1>(i)];
        E += 0.5 * pi * pi
           + pow(qi, KAPPA) / KAPPA
           + pow(abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    double qN = q[Point<1>(N - 1)];
    double pN = p[Point<1>(N - 1)];
    E += 0.5 * pN * pN
       + pow(qN, KAPPA) / KAPPA
       + 0.5 * pow(abs(qN), LAMBDA) / LAMBDA;
    return E;
}

/* ================================================================
   top_level_task
   ================================================================ */
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    /* ---- command-line parsing -------------------------------- */
    InputArgs args = Runtime::get_input_args();
    int    N     = 1024;
    int    G     = 128;
    int    steps = 100;
    double dt    = 0.01;
    for (int i = 1; i < args.argc; ++i) {
        if (!strcmp(args.argv[i], "--N")     && i + 1 < args.argc) N     = atoi(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--G")     && i + 1 < args.argc) G     = atoi(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--steps") && i + 1 < args.argc) steps = atoi(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--dt")    && i + 1 < args.argc) dt    = atof(args.argv[++i]);
    }
    int M = N / G;

    /* ---- output file ---------------------------------------- */
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }
    outfile << "Dimension: " << N
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M
            << ", steps: " << steps
            << ", dt: "    << dt << std::endl;

    /* ---- create index space & field space -------------------- */
    Rect<1> elem_rect(0, N - 1);
    IndexSpace is = runtime->create_index_space(ctx, elem_rect);
    runtime->attach_name(is, "elements_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    /* ---- create logical regions: q, p, dpdt ----------------- */
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    /* ---- colour space for M blocks -------------------------- */
    Rect<1> color_rect(0, M - 1);
    IndexSpace color_is = runtime->create_index_space(ctx, color_rect);

    /* ---- disjoint partition (block i → [i*G, (i+1)*G-1]) ---- */
    Transform<1,1> disjoint_xform;
    disjoint_xform[0][0] = G;
    IndexPartition disjoint_ip =
        runtime->create_partition_by_restriction(ctx, is, color_is,
                                                 disjoint_xform,
                                                 Rect<1>(0, G - 1),
                                                 LEGION_DISJOINT_COMPLETE_KIND);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    disjoint_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    disjoint_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, disjoint_ip);

    /* ---- ghost (aliased) partition for q force reads -------- *
     *  block 0       : [0,           G           ]              *
     *  block i (mid) : [i*G - 1,     (i+1)*G     ]              *
     *  block M-1     : [(M-1)*G - 1, N - 1       ]              *
     *                                                            *
     *  Using create_partition_by_restriction with extent [-1, G] *
     *  which is automatically clipped to the parent index space. */
    Transform<1,1> ghost_xform;
    ghost_xform[0][0] = G;
    IndexPartition ghost_ip =
        runtime->create_partition_by_restriction(ctx, is, color_is,
                                                 ghost_xform,
                                                 Rect<1>(-1, G),
                                                 LEGION_ALIASED_COMPLETE_KIND);
    LogicalPartition q_ghost_lp =
        runtime->get_logical_partition(ctx, q_lr, ghost_ip);

    /* ---- initialise q = 0, p = random ----------------------- */
    runtime->fill_field(ctx, q_lr, q_lr, FID_VAL, 0.0);
    runtime->fill_field(ctx, dpdt_lr, dpdt_lr, FID_VAL, 0.0);

    {
        /* generate the same random sequence as the HPX code */
        std::vector<double> p_init(N);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));

        InlineLauncher il(
            RegionRequirement(p_lr, READ_WRITE, EXCLUSIVE, p_lr));
        il.add_field(FID_VAL);
        PhysicalRegion p_pr = runtime->map_region(ctx, il);
        p_pr.wait_until_valid();

        const FieldAccessor<READ_WRITE, double, 1> p_acc(p_pr, FID_VAL);
        for (int i = 0; i < N; ++i)
            p_acc[Point<1>(i)] = p_init[i];

        runtime->unmap_region(ctx, p_pr);
    }

    /* ---- compute initial energy ----------------------------- */
    {
        TaskLauncher launcher(TID_ENERGY, TaskArgument(&N, sizeof(N)));
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

    /* ---- time integration loop ------------------------------ */
    Domain launch_domain(color_rect);
    ForceArgs fa; fa.N = N; fa.G = G; fa.M = M;

    for (int step = 0; step < steps; ++step) {
        for (int s = 0; s < NUM_STAGES; ++s) {

            /* -- drift: q += a[s]*dt * p ---------------------- */
            {
                DriftKickArgs da; da.alpha = coef_a[s] * dt; da.G = G;
                IndexLauncher il(TID_DRIFT, launch_domain,
                                 TaskArgument(&da, sizeof(da)),
                                 ArgumentMap());
                il.add_region_requirement(
                    RegionRequirement(q_lp, 0 /*proj*/,
                                      READ_WRITE, EXCLUSIVE, q_lr));
                il.region_requirements[0].add_field(FID_VAL);
                il.add_region_requirement(
                    RegionRequirement(p_lp, 0 /*proj*/,
                                      READ_ONLY, EXCLUSIVE, p_lr));
                il.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, il);
            }

            /* -- force: dpdt = F(q) --------------------------- */
            {
                IndexLauncher il(TID_FORCE, launch_domain,
                                 TaskArgument(&fa, sizeof(fa)),
                                 ArgumentMap());
                il.add_region_requirement(
                    RegionRequirement(q_ghost_lp, 0 /*proj*/,
                                      READ_ONLY, EXCLUSIVE, q_lr));
                il.region_requirements[0].add_field(FID_VAL);
                il.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0 /*proj*/,
                                      WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
                il.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, il);
            }

            /* -- kick: p += b[s+1]*dt * dpdt ------------------ */
            {
                DriftKickArgs da; da.alpha = coef_b[s + 1] * dt; da.G = G;
                IndexLauncher il(TID_KICK, launch_domain,
                                 TaskArgument(&da, sizeof(da)),
                                 ArgumentMap());
                il.add_region_requirement(
                    RegionRequirement(p_lp, 0 /*proj*/,
                                      READ_WRITE, EXCLUSIVE, p_lr));
                il.region_requirements[0].add_field(FID_VAL);
                il.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0 /*proj*/,
                                      READ_ONLY, EXCLUSIVE, dpdt_lr));
                il.region_requirements[1].add_field(FID_VAL);
                runtime->execute_index_space(ctx, il);
            }
        }
    }

    /* ---- compute final energy ------------------------------- */
    {
        TaskLauncher launcher(TID_ENERGY, TaskArgument(&N, sizeof(N)));
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

    outfile.close();

    /* ---- cleanup -------------------------------------------- */
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

/* ================================================================
   main – register tasks and start Legion runtime
   ================================================================ */
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TID_TOP);

    /* top-level task */
    {
        TaskVariantRegistrar reg(TID_TOP, "top_level");
        reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(reg, "top_level");
    }
    /* force task */
    {
        TaskVariantRegistrar reg(TID_FORCE, "force");
        reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        reg.set_leaf(true);
        Runtime::preregister_task_variant<force_task>(reg, "force");
    }
    /* drift task */
    {
        TaskVariantRegistrar reg(TID_DRIFT, "drift");
        reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        reg.set_leaf(true);
        Runtime::preregister_task_variant<drift_task>(reg, "drift");
    }
    /* kick task */
    {
        TaskVariantRegistrar reg(TID_KICK, "kick");
        reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        reg.set_leaf(true);
        Runtime::preregister_task_variant<kick_task>(reg, "kick");
    }
    /* energy task (returns double) */
    {
        TaskVariantRegistrar reg(TID_ENERGY, "energy");
        reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, energy_task>(reg, "energy");
    }

    return Runtime::start(argc, argv);
}
