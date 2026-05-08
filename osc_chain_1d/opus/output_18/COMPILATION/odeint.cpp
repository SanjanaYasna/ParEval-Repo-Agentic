// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

#include "legion.h"

using namespace Legion;

// ============================================================
// Physics constants (from system.hpp)
// ============================================================
static const double KAPPA  = 3.5;
static const double LAMBDA = 4.5;

namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k)
{
    if (x > 0.0)  return  checked_math::pow(x, k);
    if (x < 0.0)  return -checked_math::pow(x, k);
    return 0.0;
}

// ============================================================
// Symplectic RKN SB3A McLachlan coefficients (6 stages)
// (from boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan)
// ============================================================
static const int NUM_STAGES = 6;
static const double COEFF_A[NUM_STAGES] = {
     0.40518861839525227722,
    -0.28714404081652408900,
     0.38195542242127181178,   // 0.5 - (a0 + a1)
     0.38195542242127181178,
    -0.28714404081652408900,
     0.40518861839525227722
};
static const double COEFF_B[NUM_STAGES] = {
    -3.0 / 73.0,
     17.0 / 59.0,
     1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
     17.0 / 59.0,
    -3.0 / 73.0,
     0.0
};

// ============================================================
// Field IDs & Task IDs
// ============================================================
enum FieldIDs {
    FID_Q = 100,
    FID_P,
    FID_DPDT,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
    FORCE_TASK_ID,
    UPDATE_Q_TASK_ID,
    UPDATE_P_TASK_ID,
    ENERGY_TASK_ID,
};

// ============================================================
// Task argument structures
// ============================================================
struct ForceArgs {
    int N, G, M;
};

struct UpdateArgs {
    double factor;          // a[l]*dt  or  b[l]*dt
};

struct EnergyArgs {
    int N;
};

// ============================================================
// Force task  –  computes  dpdt = f(q)
//   Region 0 : ghost sub-region  – READ_ONLY   FID_Q
//   Region 1 : owned sub-region  – WRITE_DISCARD FID_DPDT
// ============================================================
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceArgs args = *(const ForceArgs *)task->args;
    int block = task->index_point[0];
    int G = args.G;
    int M = args.M;

    int own_lo = block * G;
    int own_hi = own_lo + G - 1;

    const FieldAccessor<READ_ONLY,     double, 1> q   (regions[0], FID_Q);
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_DPDT);

    // ---------- first block (left boundary: coupling to 0) ----------
    if (block == 0)
    {
        double coupling_lr = -signed_pow(q[own_lo], LAMBDA - 1);
        for (int i = own_lo; i < own_hi; ++i) {
            double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((double)q[i] - (double)q[i + 1], LAMBDA - 1);
            val -= coupling_lr;
            dpdt[i] = val;
        }
        if (M > 1) {
            double q_r = q[own_hi + 1];           // first elem of next block
            dpdt[own_hi] = -signed_pow(q[own_hi], KAPPA - 1)
                + coupling_lr - signed_pow((double)q[own_hi] - q_r, LAMBDA - 1);
        } else {
            // single block: right boundary coupling to 0
            dpdt[own_hi] = -signed_pow(q[own_hi], KAPPA - 1)
                + coupling_lr - signed_pow((double)q[own_hi], LAMBDA - 1);
        }
    }
    // ---------- last block (right boundary: coupling to 0) ----------
    else if (block == M - 1)
    {
        double q_l = q[own_lo - 1];               // last elem of prev block
        double coupling_lr = -signed_pow((double)q[own_lo] - q_l, LAMBDA - 1);
        for (int i = own_lo; i < own_hi; ++i) {
            double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((double)q[i] - (double)q[i + 1], LAMBDA - 1);
            val -= coupling_lr;
            dpdt[i] = val;
        }
        dpdt[own_hi] = -signed_pow(q[own_hi], KAPPA - 1)
            + coupling_lr - signed_pow((double)q[own_hi], LAMBDA - 1);
    }
    // ---------- centre blocks ----------
    else
    {
        double q_l = q[own_lo - 1];
        double q_r = q[own_hi + 1];
        double coupling_lr = -signed_pow((double)q[own_lo] - q_l, LAMBDA - 1);
        for (int i = own_lo; i < own_hi; ++i) {
            double val = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((double)q[i] - (double)q[i + 1], LAMBDA - 1);
            val -= coupling_lr;
            dpdt[i] = val;
        }
        dpdt[own_hi] = -signed_pow(q[own_hi], KAPPA - 1)
            + coupling_lr - signed_pow((double)q[own_hi] - q_r, LAMBDA - 1);
    }
}

// ============================================================
// Update-Q task :  q  +=  factor * p
//   Region 0 : owned sub-region  – READ_WRITE  FID_Q
//   Region 1 : owned sub-region  – READ_ONLY   FID_P
// ============================================================
void update_q_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    double factor = ((const UpdateArgs *)task->args)->factor;

    const FieldAccessor<READ_WRITE, double, 1> q(regions[0], FID_Q);
    const FieldAccessor<READ_ONLY,  double, 1> p(regions[1], FID_P);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); ++it)
        q[*it] = (double)q[*it] + factor * (double)p[*it];
}

// ============================================================
// Update-P task :  p  +=  factor * dpdt
//   Region 0 : owned sub-region  – READ_WRITE  FID_P
//   Region 1 : owned sub-region  – READ_ONLY   FID_DPDT
// ============================================================
void update_p_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    double factor = ((const UpdateArgs *)task->args)->factor;

    const FieldAccessor<READ_WRITE, double, 1> p   (regions[0], FID_P);
    const FieldAccessor<READ_ONLY,  double, 1> dpdt(regions[1], FID_DPDT);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); ++it)
        p[*it] = (double)p[*it] + factor * (double)dpdt[*it];
}

// ============================================================
// Energy task  –  scalar reduction over the whole region
//   Region 0 : full region  – READ_ONLY  FID_Q, FID_P
// ============================================================
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    int N = ((const EnergyArgs *)task->args)->N;

    const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_Q);
    const FieldAccessor<READ_ONLY, double, 1> p(regions[0], FID_P);

    double qi, qi1, pi;

    qi = q[0];
    double en = 0.5 * checked_math::pow(std::abs(qi), LAMBDA) / LAMBDA;

    for (int i = 0; i < N - 1; ++i) {
        qi  = q[i];
        qi1 = q[i + 1];
        pi  = p[i];
        en += 0.5 * pi * pi
            + checked_math::pow(qi, KAPPA) / KAPPA
            + checked_math::pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    qi = q[N - 1];
    pi = p[N - 1];
    en += 0.5 * pi * pi
        + checked_math::pow(qi, KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(qi), LAMBDA) / LAMBDA;

    return en;
}

// ============================================================
// Simple command-line helpers
// ============================================================
static int    get_int_arg(int argc, char **argv, const char *flag, int    def)
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
    return def;
}
static double get_dbl_arg(int argc, char **argv, const char *flag, double def)
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atof(argv[i + 1]);
    return def;
}

// ============================================================
// Top-level task
// ============================================================
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing --------------------------------
    const InputArgs &cmd = Runtime::get_input_args();
    int    N     = get_int_arg(cmd.argc, cmd.argv, "--N",     1024);
    int    G     = get_int_arg(cmd.argc, cmd.argv, "--G",     128);
    int    steps = get_int_arg(cmd.argc, cmd.argv, "--steps", 100);
    double dt    = get_dbl_arg(cmd.argc, cmd.argv, "--dt",    0.01);
    int    M     = N / G;

    assert(N > 0 && G > 0 && N % G == 0 && "N must be a positive multiple of G");
    assert(M >= 1);

    // ---- output file ----------------------------------------
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

    // ---- index space  [0 .. N-1] ----------------------------
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));
    runtime->attach_name(is, "main_is");

    // ---- field space  {Q, P, DPDT} --------------------------
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_Q);
        fa.allocate_field(sizeof(double), FID_P);
        fa.allocate_field(sizeof(double), FID_DPDT);
    }

    // ---- logical region -------------------------------------
    LogicalRegion data_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(data_lr, "data_lr");

    // ---- colour space  [0 .. M-1] ---------------------------
    IndexSpace color_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));

    // ---- disjoint partition (M blocks of G) -----------------
    IndexPartition disjoint_ip;
    {
        Transform<1, 1> transform;
        transform[0][0] = static_cast<coord_t>(G);
        Rect<1> extent(Point<1>(0), Point<1>(G - 1));
        disjoint_ip = runtime->create_partition_by_restriction(
            ctx, is, color_is, transform, extent, LEGION_DISJOINT_KIND);
    }
    LogicalPartition disjoint_lp =
        runtime->get_logical_partition(ctx, data_lr, disjoint_ip);

    // ---- ghost partition (±1 element for stencil reads) -----
    //  For color i the subspace is [i*G - 1, i*G + G] ∩ [0, N-1]
    IndexPartition ghost_ip;
    {
        Transform<1, 1> transform;
        transform[0][0] = static_cast<coord_t>(G);
        Rect<1> extent(Point<1>(-1), Point<1>(G));
        ghost_ip = runtime->create_partition_by_restriction(
            ctx, is, color_is, transform, extent, LEGION_ALIASED_KIND);
    }
    LogicalPartition ghost_lp =
        runtime->get_logical_partition(ctx, data_lr, ghost_ip);

    // ---- generate random initial momenta --------------------
    std::vector<double> p_init(N);
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::mt19937 engine(0);
        auto gen = std::bind(dist, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(gen));
    }

    // ---- initialise q=0, p=p_init, dpdt=0 (inline map) -----
    {
        InlineLauncher il(RegionRequirement(
            data_lr, WRITE_DISCARD, EXCLUSIVE, data_lr));
        il.add_field(FID_Q);
        il.add_field(FID_P);
        il.add_field(FID_DPDT);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        const FieldAccessor<WRITE_DISCARD, double, 1> qa(pr, FID_Q);
        const FieldAccessor<WRITE_DISCARD, double, 1> pa(pr, FID_P);
        const FieldAccessor<WRITE_DISCARD, double, 1> da(pr, FID_DPDT);

        for (int i = 0; i < N; ++i) {
            qa[i] = 0.0;
            pa[i] = p_init[i];
            da[i] = 0.0;
        }
        runtime->unmap_region(ctx, pr);
    }

    // ---- initial energy -------------------------------------
    {
        EnergyArgs ea; ea.N = N;
        TaskLauncher tl(ENERGY_TASK_ID,
                        TaskArgument(&ea, sizeof(ea)));
        tl.add_region_requirement(
            RegionRequirement(data_lr, READ_ONLY, EXCLUSIVE, data_lr));
        tl.region_requirements[0].add_field(FID_Q);
        tl.region_requirements[0].add_field(FID_P);
        Future f = runtime->execute_task(ctx, tl);
        double e = f.get_result<double>();
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    // ---- time integration -----------------------------------
    // symplectic_rkn_sb3a_mclachlan  (6 stages per step)
    //   for each stage l:
    //     q  +=  a[l]*dt * p
    //     dpdt = f(q)            (force)
    //     p  +=  b[l]*dt * dpdt  (skip when b[l]==0)
    // ---------------------------------------------------------
    ForceArgs  fa_args;  fa_args.N = N;  fa_args.G = G;  fa_args.M = M;
    ArgumentMap arg_map;

    for (int step = 0; step < steps; ++step) {
        for (int stage = 0; stage < NUM_STAGES; ++stage) {

            // --- update q ---
            {
                UpdateArgs ua;  ua.factor = COEFF_A[stage] * dt;
                IndexTaskLauncher il(UPDATE_Q_TASK_ID, color_is,
                    TaskArgument(&ua, sizeof(ua)), arg_map);
                // RW on Q
                il.add_region_requirement(RegionRequirement(
                    disjoint_lp, 0, READ_WRITE, EXCLUSIVE, data_lr));
                il.region_requirements[0].add_field(FID_Q);
                // RO on P
                il.add_region_requirement(RegionRequirement(
                    disjoint_lp, 0, READ_ONLY, EXCLUSIVE, data_lr));
                il.region_requirements[1].add_field(FID_P);
                runtime->execute_index_space(ctx, il);
            }

            // --- compute force ---
            {
                IndexTaskLauncher il(FORCE_TASK_ID, color_is,
                    TaskArgument(&fa_args, sizeof(fa_args)), arg_map);
                // RO on Q from ghost partition
                il.add_region_requirement(RegionRequirement(
                    ghost_lp, 0, READ_ONLY, EXCLUSIVE, data_lr));
                il.region_requirements[0].add_field(FID_Q);
                // WD on DPDT from disjoint partition
                il.add_region_requirement(RegionRequirement(
                    disjoint_lp, 0, WRITE_DISCARD, EXCLUSIVE, data_lr));
                il.region_requirements[1].add_field(FID_DPDT);
                runtime->execute_index_space(ctx, il);
            }

            // --- update p  (skip when b[stage]==0) ---
            if (COEFF_B[stage] != 0.0) {
                UpdateArgs ua;  ua.factor = COEFF_B[stage] * dt;
                IndexTaskLauncher il(UPDATE_P_TASK_ID, color_is,
                    TaskArgument(&ua, sizeof(ua)), arg_map);
                // RW on P
                il.add_region_requirement(RegionRequirement(
                    disjoint_lp, 0, READ_WRITE, EXCLUSIVE, data_lr));
                il.region_requirements[0].add_field(FID_P);
                // RO on DPDT
                il.add_region_requirement(RegionRequirement(
                    disjoint_lp, 0, READ_ONLY, EXCLUSIVE, data_lr));
                il.region_requirements[1].add_field(FID_DPDT);
                runtime->execute_index_space(ctx, il);
            }
        }
    }

    // ---- final energy ---------------------------------------
    {
        EnergyArgs ea; ea.N = N;
        TaskLauncher tl(ENERGY_TASK_ID,
                        TaskArgument(&ea, sizeof(ea)));
        tl.add_region_requirement(
            RegionRequirement(data_lr, READ_ONLY, EXCLUSIVE, data_lr));
        tl.region_requirements[0].add_field(FID_Q);
        tl.region_requirements[0].add_field(FID_P);
        Future f = runtime->execute_task(ctx, tl);
        double e = f.get_result<double>();
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(e)) << std::endl;
    }

    outfile.close();

    // ---- clean up -------------------------------------------
    runtime->destroy_logical_region(ctx, data_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ============================================================
// main  –  register tasks & start Legion runtime
// ============================================================
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    // top-level (inner – launches sub-tasks)
    {
        TaskVariantRegistrar r(TOP_LEVEL_TASK_ID, "top_level");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(r, "top_level");
    }
    // force  (leaf)
    {
        TaskVariantRegistrar r(FORCE_TASK_ID, "force");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<force_task>(r, "force");
    }
    // update_q  (leaf)
    {
        TaskVariantRegistrar r(UPDATE_Q_TASK_ID, "update_q");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<update_q_task>(r, "update_q");
    }
    // update_p  (leaf)
    {
        TaskVariantRegistrar r(UPDATE_P_TASK_ID, "update_p");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<update_p_task>(r, "update_p");
    }
    // energy  (leaf, returns double)
    {
        TaskVariantRegistrar r(ENERGY_TASK_ID, "energy");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<double, energy_task>(r, "energy");
    }

    return Runtime::start(argc, argv);
}
