// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include "legion.h"
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

/* ============================================================
   Physical constants (from system.hpp)
   ============================================================ */
const double KAPPA  = 3.5;
const double LAMBDA = 4.5;

/* ============================================================
   Symplectic RKN SB3A McLachlan coefficients (6 stages)
   (from boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan)
   ============================================================ */
static const int NUM_STAGES = 6;
static const double coeff_a[NUM_STAGES] = {
     0.40518861839525227722,
    -0.28714404081652408900,
     0.38195542242127181178,
     0.38195542242127181178,
    -0.28714404081652408900,
     0.40518861839525227722
};
static const double coeff_b[NUM_STAGES] = {
    -3.0 / 73.0,
     17.0 / 59.0,
     1.0 - 2.0 * (-3.0 / 73.0 + 17.0 / 59.0),
     17.0 / 59.0,
    -3.0 / 73.0,
     0.0
};

/* ============================================================
   Task & Field IDs
   ============================================================ */
enum { TOP_LEVEL_TASK_ID, UPDATE_TASK_ID, FORCE_TASK_ID };
enum { FID_VAL = 0 };

/* ============================================================
   Math helpers (from system.hpp)
   ============================================================ */
namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

inline double signed_pow(double x, double k)
{
    return checked_math::pow(x, k) * boost::math::sign(x);
}

/* ============================================================
   Task argument POD types
   ============================================================ */
struct UpdateArgs { double alpha; };
struct ForceArgs  { int block_type; /* 0=first 1=center 2=last 3=single */ };

/* ============================================================
   update_task :  dst[i] += alpha * src[i]
     regions[0] = dst field (READ_WRITE)
     regions[1] = src field (READ_ONLY)
   ============================================================ */
void update_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const UpdateArgs a = *reinterpret_cast<const UpdateArgs*>(task->args);

    const FieldAccessor<READ_WRITE, double, 1> dst(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> src(regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++)
        dst[*it] = dst[*it] + a.alpha * src[*it];
}

/* ============================================================
   force_task : compute dpdt = F(q) for one block
     regions[0] = dpdt subregion (WRITE_DISCARD)
     regions[1] = q   subregion (READ_ONLY)
     additional regions for ghost cells (READ_ONLY on neighbour q):
       first  (0) : regions[2] = right neighbour q
       center (1) : regions[2] = left neighbour q,
                     regions[3] = right neighbour q
       last   (2) : regions[2] = left neighbour q
       single (3) : (none)
   ============================================================ */
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    const ForceArgs fa = *reinterpret_cast<const ForceArgs*>(task->args);

    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,     double, 1> q   (regions[1], FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    const coord_t lo = rect.lo[0], hi = rect.hi[0];

    const bool has_left  = (fa.block_type == 1 || fa.block_type == 2);
    const bool has_right = (fa.block_type == 0 || fa.block_type == 1);

    double q_l = 0.0, q_r = 0.0;

    if (has_left) {
        const FieldAccessor<READ_ONLY, double, 1> ql(regions[2], FID_VAL);
        Rect<1> lr = runtime->get_index_space_domain(ctx,
            task->regions[2].region.get_index_space());
        q_l = ql[lr.hi];                       /* last element of left neighbour */
    }
    if (has_right) {
        const int ri = has_left ? 3 : 2;
        const FieldAccessor<READ_ONLY, double, 1> qr(regions[ri], FID_VAL);
        Rect<1> rr = runtime->get_index_space_domain(ctx,
            task->regions[ri].region.get_index_space());
        q_r = qr[rr.lo];                       /* first element of right neighbour */
    }

    /* left-boundary coupling */
    double clr = has_left ? -signed_pow(q[lo] - q_l, LAMBDA - 1)
                          : -signed_pow(q[lo],        LAMBDA - 1);

    /* interior elements */
    for (coord_t i = lo; i < hi; ++i) {
        double d = -signed_pow(q[i], KAPPA - 1) + clr;
        clr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] = d - clr;
    }

    /* last element */
    double d = -signed_pow(q[hi], KAPPA - 1) + clr;
    if (has_right)
        d -= signed_pow(q[hi] - q_r, LAMBDA - 1);
    else
        d -= signed_pow(q[hi], LAMBDA - 1);
    dpdt[hi] = d;
}

/* ============================================================
   Serial energy computation on inline-mapped regions
   (from system.hpp energy() functions)
   ============================================================ */
double compute_energy(const FieldAccessor<READ_ONLY, double, 1> &q,
                      const FieldAccessor<READ_ONLY, double, 1> &p,
                      size_t N)
{
    const coord_t cN = static_cast<coord_t>(N);

    double E = 0.5 * checked_math::pow(std::abs((double)q[(coord_t)0]), LAMBDA) / LAMBDA;
    for (coord_t i = 0; i < cN - 1; ++i) {
        double qi  = q[i];
        double qi1 = q[i + 1];
        double pi  = p[i];
        E += 0.5 * pi * pi
           + checked_math::pow(qi, KAPPA) / KAPPA
           + checked_math::pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
    }
    double qn = q[cN - 1];
    double pn = p[cN - 1];
    E += 0.5 * pn * pn
       + checked_math::pow(qn, KAPPA) / KAPPA
       + 0.5 * checked_math::pow(std::abs(qn), LAMBDA) / LAMBDA;
    return E;
}

/* ============================================================
   Helper: inline-map a region, compute energy, unmap
   ============================================================ */
double inline_energy(LogicalRegion q_lr, LogicalRegion p_lr,
                     size_t N, Context ctx, Runtime *runtime)
{
    InlineLauncher il_q(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    il_q.add_field(FID_VAL);
    InlineLauncher il_p(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    il_p.add_field(FID_VAL);

    PhysicalRegion pr_q = runtime->map_region(ctx, il_q);
    PhysicalRegion pr_p = runtime->map_region(ctx, il_p);
    pr_q.wait_until_valid();
    pr_p.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> qa(pr_q, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> pa(pr_p, FID_VAL);
    double E = compute_energy(qa, pa, N);

    runtime->unmap_region(ctx, pr_q);
    runtime->unmap_region(ctx, pr_p);
    return E;
}

/* ============================================================
   Helper: launch force tasks for all M blocks
   ============================================================ */
void launch_force(size_t M,
                  LogicalPartition dpdt_lp, LogicalRegion dpdt_lr,
                  LogicalPartition q_lp,    LogicalRegion q_lr,
                  Context ctx, Runtime *runtime)
{
    for (size_t i = 0; i < M; ++i) {
        ForceArgs fa;
        if (M == 1)          fa.block_type = 3;
        else if (i == 0)     fa.block_type = 0;
        else if (i == M - 1) fa.block_type = 2;
        else                 fa.block_type = 1;

        TaskLauncher tl(FORCE_TASK_ID,
                        TaskArgument(&fa, sizeof(fa)));

        LogicalRegion dsub = runtime->get_logical_subregion_by_color(
            ctx, dpdt_lp, DomainPoint(Point<1>(static_cast<coord_t>(i))));
        LogicalRegion qsub = runtime->get_logical_subregion_by_color(
            ctx, q_lp, DomainPoint(Point<1>(static_cast<coord_t>(i))));

        /* regions[0]: own dpdt (WRITE_DISCARD) */
        tl.add_region_requirement(
            RegionRequirement(dsub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        tl.region_requirements[0].add_field(FID_VAL);

        /* regions[1]: own q (READ_ONLY) */
        tl.add_region_requirement(
            RegionRequirement(qsub, READ_ONLY, EXCLUSIVE, q_lr));
        tl.region_requirements[1].add_field(FID_VAL);

        if (fa.block_type == 0) {
            /* first block: right neighbour */
            LogicalRegion qr = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(static_cast<coord_t>(i + 1))));
            tl.add_region_requirement(
                RegionRequirement(qr, READ_ONLY, EXCLUSIVE, q_lr));
            tl.region_requirements[2].add_field(FID_VAL);
        } else if (fa.block_type == 2) {
            /* last block: left neighbour */
            LogicalRegion ql = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(static_cast<coord_t>(i - 1))));
            tl.add_region_requirement(
                RegionRequirement(ql, READ_ONLY, EXCLUSIVE, q_lr));
            tl.region_requirements[2].add_field(FID_VAL);
        } else if (fa.block_type == 1) {
            /* centre block: left then right neighbour */
            LogicalRegion ql = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(static_cast<coord_t>(i - 1))));
            tl.add_region_requirement(
                RegionRequirement(ql, READ_ONLY, EXCLUSIVE, q_lr));
            tl.region_requirements[2].add_field(FID_VAL);

            LogicalRegion qr = runtime->get_logical_subregion_by_color(
                ctx, q_lp, DomainPoint(Point<1>(static_cast<coord_t>(i + 1))));
            tl.add_region_requirement(
                RegionRequirement(qr, READ_ONLY, EXCLUSIVE, q_lr));
            tl.region_requirements[3].add_field(FID_VAL);
        }
        /* single (3): no extra regions */

        runtime->execute_task(ctx, tl);
    }
}

/* ============================================================
   top_level_task  –  orchestrates the whole computation
   ============================================================ */
void top_level_task(const Task * /*task*/,
                    const std::vector<PhysicalRegion> & /*regions*/,
                    Context ctx, Runtime *runtime)
{
    /* ---------- command-line parsing ---------- */
    size_t N = 1024, G = 128, steps = 100;
    double dt = 0.01;
    {
        const InputArgs &ia = Runtime::get_input_args();
        for (int i = 1; i < ia.argc; ++i) {
            if (!strcmp(ia.argv[i], "--N") && i + 1 < ia.argc)
                N = static_cast<size_t>(std::atoi(ia.argv[++i]));
            else if (!strcmp(ia.argv[i], "--G") && i + 1 < ia.argc)
                G = static_cast<size_t>(std::atoi(ia.argv[++i]));
            else if (!strcmp(ia.argv[i], "--steps") && i + 1 < ia.argc)
                steps = static_cast<size_t>(std::atoi(ia.argv[++i]));
            else if (!strcmp(ia.argv[i], "--dt") && i + 1 < ia.argc)
                dt = std::atof(ia.argv[++i]);
        }
    }
    const size_t M = N / G;

    /* ---------- output file ---------- */
    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }
    outfile << "Dimension: " << N
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: "  << M
            << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    /* ---------- create index space & field space ---------- */
    const Rect<1> elem_rect(0, static_cast<coord_t>(N - 1));
    IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    /* Three logical regions: q, p, dpdt */
    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    /* ---------- equal partition into M blocks ---------- */
    const Rect<1> color_rect(0, static_cast<coord_t>(M - 1));
    IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_rect);
    IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    /* ---------- initialisation (inline mapping) ---------- */
    {
        /* q = 0 */
        InlineLauncher il_q(RegionRequirement(q_lr, WRITE_DISCARD,
                                              EXCLUSIVE, q_lr));
        il_q.add_field(FID_VAL);
        PhysicalRegion pr_q = runtime->map_region(ctx, il_q);
        pr_q.wait_until_valid();
        {
            const FieldAccessor<WRITE_DISCARD, double, 1> qa(pr_q, FID_VAL);
            for (coord_t i = 0; i < static_cast<coord_t>(N); ++i)
                qa[i] = 0.0;
        }
        runtime->unmap_region(ctx, pr_q);

        /* p = random [-1,1], seeded with 0 (same as original) */
        InlineLauncher il_p(RegionRequirement(p_lr, WRITE_DISCARD,
                                              EXCLUSIVE, p_lr));
        il_p.add_field(FID_VAL);
        PhysicalRegion pr_p = runtime->map_region(ctx, il_p);
        pr_p.wait_until_valid();
        {
            const FieldAccessor<WRITE_DISCARD, double, 1> pa(pr_p, FID_VAL);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            std::mt19937 engine(0);
            auto gen = std::bind(dist, engine);
            for (coord_t i = 0; i < static_cast<coord_t>(N); ++i)
                pa[i] = gen();
        }
        runtime->unmap_region(ctx, pr_p);
    }

    /* ---------- initial energy ---------- */
    {
        double E = inline_energy(q_lr, p_lr, N, ctx, runtime);
        outfile << "Initialization complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
    }

    /* ==========================================================
       Time integration – manual symplectic_rkn_sb3a_mclachlan
       Each step has NUM_STAGES stages.  For stage s:
         1) q += coeff_a[s] * dt * p           (index launch)
         2) dpdt = F(q)                         (individual launches)
         3) p += coeff_b[s] * dt * dpdt         (index launch)
       Steps 2-3 are skipped when coeff_b[s] == 0.
       ========================================================== */
    for (size_t step = 0; step < steps; ++step) {
        for (int s = 0; s < NUM_STAGES; ++s) {

            /* --- 1) update q: q += a*dt * p --- */
            if (coeff_a[s] != 0.0) {
                UpdateArgs ua;
                ua.alpha = coeff_a[s] * dt;

                IndexLauncher il(UPDATE_TASK_ID, color_is,
                    TaskArgument(&ua, sizeof(ua)), ArgumentMap());

                il.add_region_requirement(
                    RegionRequirement(q_lp, 0/*proj*/, READ_WRITE,
                                      EXCLUSIVE, q_lr));
                il.region_requirements[0].add_field(FID_VAL);

                il.add_region_requirement(
                    RegionRequirement(p_lp, 0/*proj*/, READ_ONLY,
                                      EXCLUSIVE, p_lr));
                il.region_requirements[1].add_field(FID_VAL);

                runtime->execute_index_space(ctx, il);
            }

            /* --- 2-3) force + p update --- */
            if (coeff_b[s] != 0.0) {
                /* 2) compute dpdt = F(q) */
                launch_force(M, dpdt_lp, dpdt_lr, q_lp, q_lr, ctx, runtime);

                /* 3) update p: p += b*dt * dpdt */
                UpdateArgs ua;
                ua.alpha = coeff_b[s] * dt;

                IndexLauncher il(UPDATE_TASK_ID, color_is,
                    TaskArgument(&ua, sizeof(ua)), ArgumentMap());

                il.add_region_requirement(
                    RegionRequirement(p_lp, 0/*proj*/, READ_WRITE,
                                      EXCLUSIVE, p_lr));
                il.region_requirements[0].add_field(FID_VAL);

                il.add_region_requirement(
                    RegionRequirement(dpdt_lp, 0/*proj*/, READ_ONLY,
                                      EXCLUSIVE, dpdt_lr));
                il.region_requirements[1].add_field(FID_VAL);

                runtime->execute_index_space(ctx, il);
            }
        }
    }

    /* ---------- final energy ---------- */
    {
        double E = inline_energy(q_lr, p_lr, N, ctx, runtime);
        outfile << "Integration complete, energy: "
                << static_cast<long long>(std::round(E)) << std::endl;
    }

    /* ---------- cleanup ---------- */
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

/* ============================================================
   main – register tasks & start Legion runtime
   ============================================================ */
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar r(TOP_LEVEL_TASK_ID, "top_level");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(r, "top_level");
    }
    {
        TaskVariantRegistrar r(UPDATE_TASK_ID, "update");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf();
        Runtime::preregister_task_variant<update_task>(r, "update");
    }
    {
        TaskVariantRegistrar r(FORCE_TASK_ID, "force");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf();
        Runtime::preregister_task_variant<force_task>(r, "force");
    }

    return Runtime::start(argc, argv);
}
