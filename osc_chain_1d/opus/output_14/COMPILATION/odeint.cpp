// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ============================================================
// Constants (from system.hpp)
// ============================================================
static const double KAPPA = 3.5;
static const double LAMBDA_EXP = 4.5;

// ============================================================
// Stepper coefficients for symplectic_rkn_sb3a_mclachlan
// Extracted from boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp
// ============================================================
static const int NUM_STAGES = 6;

struct StepperCoefficients {
    double a[6];
    double b[6];
};

static StepperCoefficients make_stepper_coefficients() {
    StepperCoefficients c;
    // These literal values match the Boost header exactly
    const double v1 = 0.40518861839525227722;
    const double v2 = 0.28714404081652408900; // unsigned
    c.a[0] =  v1;
    c.a[1] = -v2;
    c.a[2] =  0.5 - (v1 - v2);   // = 0.38195542242127181178
    c.a[3] =  c.a[2];
    c.a[4] =  c.a[1];
    c.a[5] =  c.a[0];

    // b coefficients directly from Boost header
    c.b[0] = 0.0;
    c.b[1] = 1.0 / (2.0 - 2.0 * v1);
    c.b[2] = 1.0 - 1.0 / (1.0 - v1);
    c.b[3] = c.b[2];
    c.b[4] = c.b[1];
    c.b[5] = 0.0;
    return c;
}

// ============================================================
// IDs
// ============================================================
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 1,
    INIT_ZERO_TASK_ID,
    INIT_DATA_TASK_ID,
    FORCE_TASK_ID,
    UPDATE_TASK_ID,
};

enum FieldIDs {
    FID_VAL = 101,
};

// ============================================================
// Math helpers (from system.hpp)
// ============================================================
namespace checked_math {
    inline double pow(double x, double y) {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

static inline double signed_pow(double x, double k) {
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

// ============================================================
// Task argument structures
// ============================================================
struct ForceArgs {
    int block_idx;
    int M;
    int G;
    bool has_left;
    bool has_right;
};

struct UpdateArgs {
    double alpha1;
    double alpha2;
};

// ============================================================
// INIT ZERO TASK – fills a sub-region with 0.0
// Region 0: target (WRITE_DISCARD)
// ============================================================
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

// ============================================================
// INIT DATA TASK – copies doubles from task argument into region
// Region 0: target (WRITE_DISCARD)
// ============================================================
void init_data_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    const double *data = reinterpret_cast<const double*>(task->args);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    int idx = 0;
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        acc[*pir] = data[idx++];
}

// ============================================================
// FORCE TASK – computes dpdt = f(q) for one block
// Region 0 : own q        (READ_ONLY)
// Region 1 : own dpdt     (WRITE_DISCARD)
// Region 2 : left-neighbor q  (READ_ONLY)  – present only if has_left
// Region 2/3 : right-neighbor q (READ_ONLY) – present only if has_right
// ============================================================
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceArgs args = *reinterpret_cast<const ForceArgs*>(task->args);
    const int bi = args.block_idx;
    const int M  = args.M;
    const int G  = args.G;

    // ---- read own q into local buffer ----
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    Domain own_dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> own_rect(own_dom);

    std::vector<double> q(G);
    {
        int idx = 0;
        for (PointInRectIterator<1> pir(own_rect); pir(); pir++)
            q[idx++] = q_acc[*pir];
    }

    // ---- read boundary values from neighbors ----
    double q_left_last   = 0.0;
    double q_right_first = 0.0;
    int reg_idx = 2;

    if (args.has_left) {
        const FieldAccessor<READ_ONLY, double, 1> ql_acc(regions[reg_idx], FID_VAL);
        Domain ld = runtime->get_index_space_domain(ctx,
            task->regions[reg_idx].region.get_index_space());
        Rect<1> lr(ld);
        // last element of left neighbor
        for (PointInRectIterator<1> pir(lr); pir(); pir++)
            q_left_last = ql_acc[*pir];
        reg_idx++;
    }
    if (args.has_right) {
        const FieldAccessor<READ_ONLY, double, 1> qr_acc(regions[reg_idx], FID_VAL);
        Domain rd = runtime->get_index_space_domain(ctx,
            task->regions[reg_idx].region.get_index_space());
        Rect<1> rr(rd);
        PointInRectIterator<1> pir(rr);
        q_right_first = qr_acc[*pir];
    }

    // ---- compute forces (mirroring system.hpp) ----
    std::vector<double> dpdt(G, 0.0);

    if (bi == 0) {
        // first block
        double clr = -signed_pow(q[0], LAMBDA_EXP - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + clr;
            clr = signed_pow(q[i] - q[i + 1], LAMBDA_EXP - 1);
            dpdt[i] -= clr;
        }
        dpdt[G-1] = -signed_pow(q[G-1], KAPPA - 1) + clr
                     - signed_pow(q[G-1] - q_right_first, LAMBDA_EXP - 1);
    } else if (bi == M - 1) {
        // last block
        double clr = -signed_pow(q[0] - q_left_last, LAMBDA_EXP - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + clr;
            clr = signed_pow(q[i] - q[i + 1], LAMBDA_EXP - 1);
            dpdt[i] -= clr;
        }
        dpdt[G-1] = -signed_pow(q[G-1], KAPPA - 1) + clr
                     - signed_pow(q[G-1], LAMBDA_EXP - 1);
    } else {
        // center block
        double clr = -signed_pow(q[0] - q_left_last, LAMBDA_EXP - 1);
        for (int i = 0; i < G - 1; i++) {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1) + clr;
            clr = signed_pow(q[i] - q[i + 1], LAMBDA_EXP - 1);
            dpdt[i] -= clr;
        }
        dpdt[G-1] = -signed_pow(q[G-1], KAPPA - 1) + clr
                     - signed_pow(q[G-1] - q_right_first, LAMBDA_EXP - 1);
    }

    // ---- write dpdt ----
    const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);
    Domain dpdt_dom = runtime->get_index_space_domain(ctx,
        task->regions[1].region.get_index_space());
    Rect<1> dpdt_rect(dpdt_dom);
    {
        int idx = 0;
        for (PointInRectIterator<1> pir(dpdt_rect); pir(); pir++)
            dpdt_acc[*pir] = dpdt[idx++];
    }
}

// ============================================================
// UPDATE TASK – dest[i] = alpha1*dest[i] + alpha2*src[i]
// Region 0 : dest (READ_WRITE)
// Region 1 : src  (READ_ONLY)
// ============================================================
void update_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    UpdateArgs args = *reinterpret_cast<const UpdateArgs*>(task->args);
    const FieldAccessor<READ_WRITE, double, 1> dst(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> src(regions[1], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect(dom);
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
        dst[*pir] = args.alpha1 * dst[*pir] + args.alpha2 * src[*pir];
}

// ============================================================
// Helper: get a sub-region by color
// ============================================================
static inline LogicalRegion sub(Runtime *rt, Context ctx,
                                LogicalPartition lp, int color)
{
    return rt->get_logical_subregion_by_color(ctx, lp, color);
}

// ============================================================
// Helper: launch M update tasks  dest = alpha1*dest + alpha2*src
// ============================================================
static void launch_updates(Runtime *runtime, Context ctx,
                           LogicalRegion dest_lr, LogicalPartition dest_lp,
                           LogicalRegion src_lr,  LogicalPartition src_lp,
                           int M, double alpha1, double alpha2)
{
    UpdateArgs ua;
    ua.alpha1 = alpha1;
    ua.alpha2 = alpha2;

    for (int i = 0; i < M; i++) {
        TaskLauncher launcher(UPDATE_TASK_ID,
                              TaskArgument(&ua, sizeof(ua)));
        launcher.add_region_requirement(
            RegionRequirement(sub(runtime, ctx, dest_lp, i),
                              READ_WRITE, EXCLUSIVE, dest_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        launcher.add_region_requirement(
            RegionRequirement(sub(runtime, ctx, src_lp, i),
                              READ_ONLY, EXCLUSIVE, src_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        runtime->execute_task(ctx, launcher);
    }
}

// ============================================================
// Helper: launch M force tasks  dpdt = f(q)
// ============================================================
static void launch_forces(Runtime *runtime, Context ctx,
                          LogicalRegion q_lr,    LogicalPartition q_lp,
                          LogicalRegion dpdt_lr, LogicalPartition dpdt_lp,
                          int M, int G)
{
    for (int i = 0; i < M; i++) {
        ForceArgs fa;
        fa.block_idx = i;
        fa.M = M;
        fa.G = G;
        fa.has_left  = (i > 0);
        fa.has_right = (i < M - 1);

        TaskLauncher launcher(FORCE_TASK_ID,
                              TaskArgument(&fa, sizeof(fa)));

        // own q
        launcher.add_region_requirement(
            RegionRequirement(sub(runtime, ctx, q_lp, i),
                              READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // own dpdt
        launcher.add_region_requirement(
            RegionRequirement(sub(runtime, ctx, dpdt_lp, i),
                              WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // left neighbor q
        if (fa.has_left) {
            launcher.add_region_requirement(
                RegionRequirement(sub(runtime, ctx, q_lp, i - 1),
                                  READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }
        // right neighbor q
        if (fa.has_right) {
            launcher.add_region_requirement(
                RegionRequirement(sub(runtime, ctx, q_lp, i + 1),
                                  READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        runtime->execute_task(ctx, launcher);
    }
}

// ============================================================
// Helper: compute total energy (inline-mapped in caller)
// ============================================================
static double compute_energy(Runtime *runtime, Context ctx,
                             LogicalRegion q_lr, LogicalRegion p_lr,
                             int N)
{
    RegionRequirement q_req(q_lr, READ_ONLY, EXCLUSIVE, q_lr);
    q_req.add_field(FID_VAL);
    PhysicalRegion q_pr = runtime->map_region(ctx, q_req);
    q_pr.wait_until_valid();

    RegionRequirement p_req(p_lr, READ_ONLY, EXCLUSIVE, p_lr);
    p_req.add_field(FID_VAL);
    PhysicalRegion p_pr = runtime->map_region(ctx, p_req);
    p_pr.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> qa(q_pr, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> pa(p_pr, FID_VAL);

    std::vector<double> q(N), p(N);
    for (int i = 0; i < N; i++) {
        q[i] = qa[i];
        p[i] = pa[i];
    }

    runtime->unmap_region(ctx, q_pr);
    runtime->unmap_region(ctx, p_pr);

    // energy formula from system.hpp
    double en = 0.5 * checked_math::pow(std::abs(q[0]), LAMBDA_EXP) / LAMBDA_EXP;
    for (int i = 0; i < N - 1; i++) {
        en += 0.5 * p[i] * p[i]
            + checked_math::pow(q[i], KAPPA) / KAPPA
            + checked_math::pow(std::abs(q[i] - q[i + 1]), LAMBDA_EXP) / LAMBDA_EXP;
    }
    en += 0.5 * p[N-1] * p[N-1]
        + checked_math::pow(q[N-1], KAPPA) / KAPPA
        + 0.5 * checked_math::pow(std::abs(q[N-1]), LAMBDA_EXP) / LAMBDA_EXP;
    return en;
}

// ============================================================
// TOP-LEVEL TASK
// ============================================================
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing ----
    std::size_t N     = 1024;
    std::size_t G     = 128;
    std::size_t steps = 100;
    double      dt    = 0.01;

    const InputArgs &args = Runtime::get_input_args();
    for (int i = 1; i < args.argc; i++) {
        if (!strcmp(args.argv[i], "--N") && i + 1 < args.argc)
            N = std::atol(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--G") && i + 1 < args.argc)
            G = std::atol(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--steps") && i + 1 < args.argc)
            steps = std::atol(args.argv[++i]);
        else if (!strcmp(args.argv[i], "--dt") && i + 1 < args.argc)
            dt = std::atof(args.argv[++i]);
    }

    const std::size_t M = N / G;
    assert(M * G == N && "N must be divisible by G");

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

    // ---- create index / field spaces ----
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, (coord_t)(N - 1)));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);

    // ---- partition into M blocks ----
    IndexSpace color_is = runtime->create_index_space(ctx, Rect<1>(0, (coord_t)(M - 1)));
    IndexPartition ip    = runtime->create_equal_partition(ctx, is, color_is);

    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr, ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr, ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

    // ---- generate initial p data (identical RNG to HPX version) ----
    std::vector<double> p_init(N);
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::mt19937 engine(0);
        auto generator = std::bind(distribution, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(generator));
    }

    // ---- initialize q = 0, p = p_init per block ----
    for (std::size_t i = 0; i < M; i++) {
        // q block -> zero
        {
            TaskLauncher launcher(INIT_ZERO_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(sub(runtime, ctx, q_lp, (int)i),
                                  WRITE_DISCARD, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // p block -> copy from p_init
        {
            TaskLauncher launcher(INIT_DATA_TASK_ID,
                TaskArgument(p_init.data() + i * G, G * sizeof(double)));
            launcher.add_region_requirement(
                RegionRequirement(sub(runtime, ctx, p_lp, (int)i),
                                  WRITE_DISCARD, EXCLUSIVE, p_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
        // dpdt block -> zero (scratch)
        {
            TaskLauncher launcher(INIT_ZERO_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(sub(runtime, ctx, dpdt_lp, (int)i),
                                  WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }

    // ---- compute initial energy ----
    double e0 = compute_energy(runtime, ctx, q_lr, p_lr, (int)N);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(e0)) << std::endl;

    // ---- stepper coefficients ----
    StepperCoefficients sc = make_stepper_coefficients();

    // ---- time integration (symplectic_rkn_sb3a_mclachlan) ----
    //
    // Algorithm from symplectic_nystroem_stepper_base::do_step_impl:
    //   for l = 0 .. 5:
    //       if l > 0:  q = q + coef_b[l]*dt * p
    //       dpdt = system(q)
    //       p = p + coef_a[l]*dt * dpdt
    //   q = q + coef_b[0]*dt * p          (no-op since coef_b[0]=0)
    //
    for (std::size_t step = 0; step < steps; step++) {
        for (int l = 0; l < NUM_STAGES; l++) {
            // ---- position half-step ----
            if (l > 0 && sc.b[l] != 0.0) {
                launch_updates(runtime, ctx,
                               q_lr, q_lp,   // dest = q
                               p_lr, p_lp,   // src  = p
                               (int)M, 1.0, sc.b[l] * dt);
            }
            // ---- force evaluation ----
            launch_forces(runtime, ctx,
                          q_lr, q_lp, dpdt_lr, dpdt_lp,
                          (int)M, (int)G);

            // ---- momentum half-step ----
            launch_updates(runtime, ctx,
                           p_lr, p_lp,       // dest = p
                           dpdt_lr, dpdt_lp, // src  = dpdt
                           (int)M, 1.0, sc.a[l] * dt);
        }
        // final position update (coef_b[0] == 0 → no-op)
        if (sc.b[0] != 0.0) {
            launch_updates(runtime, ctx,
                           q_lr, q_lp, p_lr, p_lp,
                           (int)M, 1.0, sc.b[0] * dt);
        }
    }

    // ---- compute final energy ----
    double e1 = compute_energy(runtime, ctx, q_lr, p_lr, (int)N);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(e1)) << std::endl;

    outfile.close();

    // ---- cleanup ----
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
    runtime->destroy_index_space(ctx, color_is);
}

// ============================================================
// main – register tasks & start Legion runtime
// ============================================================
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar r(TOP_LEVEL_TASK_ID, "top_level");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(r, "top_level");
    }
    {
        TaskVariantRegistrar r(INIT_ZERO_TASK_ID, "init_zero");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<init_zero_task>(r, "init_zero");
    }
    {
        TaskVariantRegistrar r(INIT_DATA_TASK_ID, "init_data");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<init_data_task>(r, "init_data");
    }
    {
        TaskVariantRegistrar r(FORCE_TASK_ID, "force");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<force_task>(r, "force");
    }
    {
        TaskVariantRegistrar r(UPDATE_TASK_ID, "update");
        r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        r.set_leaf(true);
        Runtime::preregister_task_variant<update_task>(r, "update");
    }

    return Runtime::start(argc, argv);
}
