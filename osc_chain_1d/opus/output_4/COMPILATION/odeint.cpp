// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <functional>
#include <cstring>
#include <utility>
#include <cassert>

#include "legion.h"

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// ---------------------------------------------------------------------------
// Physical constants (from system.hpp)
// ---------------------------------------------------------------------------
static const double KAPPA  = 3.5;
static const double LAMBDA = 4.5;

// ---------------------------------------------------------------------------
// Math helpers (from system.hpp)
// ---------------------------------------------------------------------------
namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0) return 0.0;
        return std::pow(std::abs(x), y);
    }
}

static double signed_pow(double x, double k)
{
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
typedef std::vector<double> dvec;

// ---------------------------------------------------------------------------
// Task IDs
// ---------------------------------------------------------------------------
enum {
    TOP_LEVEL_TASK_ID,
    INIT_Q_TASK_ID,
    INIT_P_TASK_ID,
    FORCE_TASK_ID,
    SCALE_SUM_TASK_ID,
    ENERGY_TASK_ID,
};

// ---------------------------------------------------------------------------
// Field IDs
// ---------------------------------------------------------------------------
enum {
    FID_VAL,
};

// ---------------------------------------------------------------------------
// Argument structs serialised into TaskArgument
// ---------------------------------------------------------------------------
struct InitPArgs {
    size_t offset; // start index into global p_init
    size_t count;  // number of elements
};

struct ForceArgs {
    int    block_idx;
    int    num_blocks;
};

struct ScaleSumArgs {
    double alpha1;
    double alpha2;
};

// ---------------------------------------------------------------------------
// INIT_Q_TASK – fill a sub-region of q with zeros
// ---------------------------------------------------------------------------
void init_q_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    for (PointInRectIterator<1> it(rect); it(); it++)
        acc[*it] = 0.0;
}

// ---------------------------------------------------------------------------
// INIT_P_TASK – copy a slice of the global p_init into a sub-region
// ---------------------------------------------------------------------------
void init_p_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
    const char *buf = static_cast<const char *>(task->args);
    InitPArgs hdr;
    std::memcpy(&hdr, buf, sizeof(hdr));
    const double *data = reinterpret_cast<const double *>(buf + sizeof(hdr));

    const FieldAccessor<WRITE_DISCARD, double, 1> acc(regions[0], FID_VAL);
    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    size_t j = 0;
    for (PointInRectIterator<1> it(rect); it(); it++, j++)
        acc[*it] = data[hdr.offset + j];
}

// ---------------------------------------------------------------------------
// FORCE_TASK – compute dp/dt for one block
//   Region 0: q sub-region (READ_ONLY)
//   Region 1: dpdt sub-region (WRITE_DISCARD)
//   Region 2 (if not first): left neighbour q sub-region (READ_ONLY)
//   Region 2/3 (if not last): right neighbour q sub-region (READ_ONLY)
// ---------------------------------------------------------------------------
void force_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
    ForceArgs args;
    assert(task->arglen == sizeof(ForceArgs));
    std::memcpy(&args, task->args, sizeof(args));

    const FieldAccessor<READ_ONLY,     double, 1> q_acc(regions[0],  FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> dp_acc(regions[1], FID_VAL);

    Domain dom_q = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect_q = dom_q;
    const coord_t lo = rect_q.lo[0];
    const coord_t hi = rect_q.hi[0];
    const size_t  N  = static_cast<size_t>(hi - lo + 1);

    // Read block into a local vector for easy indexing
    std::vector<double> q(N);
    {
        size_t j = 0;
        for (PointInRectIterator<1> it(rect_q); it(); it++, j++)
            q[j] = q_acc[*it];
    }

    std::vector<double> dpdt(N);

    bool is_first = (args.block_idx == 0);
    bool is_last  = (args.block_idx == args.num_blocks - 1);

    // Read ghost values from neighbour sub-regions
    double left_ghost = 0.0;
    double right_ghost = 0.0;

    int reg_idx = 2;
    if (!is_first) {
        const FieldAccessor<READ_ONLY, double, 1> left_acc(regions[reg_idx], FID_VAL);
        Rect<1> rect_left = runtime->get_index_space_domain(ctx,
            task->regions[reg_idx].region.get_index_space());
        left_ghost = left_acc[rect_left.hi[0]]; // last element of left neighbour
        reg_idx++;
    }
    if (!is_last) {
        const FieldAccessor<READ_ONLY, double, 1> right_acc(regions[reg_idx], FID_VAL);
        Rect<1> rect_right = runtime->get_index_space_domain(ctx,
            task->regions[reg_idx].region.get_index_space());
        right_ghost = right_acc[rect_right.lo[0]]; // first element of right neighbour
    }

    // Compute forces
    double coupling_lr;
    if (is_first)
        coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    else
        coupling_lr = -signed_pow(q[0] - left_ghost, LAMBDA - 1);

    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i]    = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i]   -= coupling_lr;
    }

    if (is_last) {
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                      + coupling_lr
                      - signed_pow(q[N - 1], LAMBDA - 1);
    } else {
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                      + coupling_lr
                      - signed_pow(q[N - 1] - right_ghost, LAMBDA - 1);
    }

    // Write back
    {
        Domain dom_dp = runtime->get_index_space_domain(ctx,
            task->regions[1].region.get_index_space());
        Rect<1> rect_dp = dom_dp;
        size_t j = 0;
        for (PointInRectIterator<1> it(rect_dp); it(); it++, j++)
            dp_acc[*it] = dpdt[j];
    }
}

// ---------------------------------------------------------------------------
// SCALE_SUM_TASK – s1[i] = alpha1*s1[i] + alpha2*s2[i]
//   Region 0: s1 sub-region (READ_WRITE)
//   Region 1: s2 sub-region (READ_ONLY)
// ---------------------------------------------------------------------------
void scale_sum_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    ScaleSumArgs args;
    assert(task->arglen == sizeof(ScaleSumArgs));
    std::memcpy(&args, task->args, sizeof(args));

    const FieldAccessor<READ_WRITE, double, 1> s1(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY,  double, 1> s2(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    for (PointInRectIterator<1> it(rect); it(); it++)
        s1[*it] = args.alpha1 * s1[*it] + args.alpha2 * s2[*it];
}

// ---------------------------------------------------------------------------
// ENERGY_TASK – compute total energy (reduction to a Future<double>)
//   Region 0: full q (READ_ONLY)
//   Region 1: full p (READ_ONLY)
// ---------------------------------------------------------------------------
double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_VAL);

    Domain dom = runtime->get_index_space_domain(ctx,
        task->regions[0].region.get_index_space());
    Rect<1> rect = dom;
    const size_t N = static_cast<size_t>(rect.hi[0] - rect.lo[0] + 1);
    const coord_t base = rect.lo[0];

    dvec q(N), p(N);
    for (size_t i = 0; i < N; ++i) {
        q[i] = q_acc[base + (coord_t)i];
        p[i] = p_acc[base + (coord_t)i];
    }

    using checked_math::pow;
    double e = 0.5 * pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i]
           + pow(q[i], KAPPA) / KAPPA
           + pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

// ---------------------------------------------------------------------------
// Helper: launch index-space scale_sum   s1 = alpha1*s1 + alpha2*s2
// ---------------------------------------------------------------------------
static void launch_scale_sum(Runtime *runtime, Context ctx,
                             LogicalRegion s1_lr, LogicalPartition s1_lp,
                             LogicalRegion s2_lr, LogicalPartition s2_lp,
                             IndexSpace block_is,
                             double alpha1, double alpha2)
{
    ScaleSumArgs args;
    args.alpha1 = alpha1;
    args.alpha2 = alpha2;

    IndexLauncher launcher(SCALE_SUM_TASK_ID, block_is,
                           TaskArgument(&args, sizeof(args)),
                           ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(s1_lp, 0,
                          READ_WRITE, EXCLUSIVE, s1_lr));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(s2_lp, 0, READ_ONLY, EXCLUSIVE, s2_lr));
    launcher.region_requirements[1].add_field(FID_VAL);

    runtime->execute_index_space(ctx, launcher);
}

// ---------------------------------------------------------------------------
// Helper: launch force computation across M blocks
//   Each task reads its own q sub-region plus neighbour sub-regions directly.
//   No inline mapping needed – Legion manages all dependencies.
// ---------------------------------------------------------------------------
static void launch_force(Runtime *runtime, Context ctx,
                         LogicalRegion q_lr,  LogicalPartition q_lp,
                         LogicalRegion dp_lr, LogicalPartition dp_lp,
                         IndexSpace block_is, size_t M, size_t G, size_t N)
{
    std::vector<Future> futures(M);
    for (size_t b = 0; b < M; ++b) {
        ForceArgs fa;
        fa.block_idx   = static_cast<int>(b);
        fa.num_blocks  = static_cast<int>(M);

        LogicalRegion q_sub  = runtime->get_logical_subregion_by_color(ctx, q_lp,
                                   static_cast<coord_t>(b));
        LogicalRegion dp_sub = runtime->get_logical_subregion_by_color(ctx, dp_lp,
                                   static_cast<coord_t>(b));

        TaskLauncher launcher(FORCE_TASK_ID,
                              TaskArgument(&fa, sizeof(fa)));

        // Region 0: own q sub-region
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 1: own dpdt sub-region
        launcher.add_region_requirement(
            RegionRequirement(dp_sub, WRITE_DISCARD, EXCLUSIVE, dp_lr));
        launcher.region_requirements.back().add_field(FID_VAL);

        // Region 2: left neighbour q (if not first block)
        if (b > 0) {
            LogicalRegion left_sub = runtime->get_logical_subregion_by_color(
                ctx, q_lp, static_cast<coord_t>(b - 1));
            launcher.add_region_requirement(
                RegionRequirement(left_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        // Region 2/3: right neighbour q (if not last block)
        if (b < M - 1) {
            LogicalRegion right_sub = runtime->get_logical_subregion_by_color(
                ctx, q_lp, static_cast<coord_t>(b + 1));
            launcher.add_region_requirement(
                RegionRequirement(right_sub, READ_ONLY, EXCLUSIVE, q_lr));
            launcher.region_requirements.back().add_field(FID_VAL);
        }

        futures[b] = runtime->execute_task(ctx, launcher);
    }
    // Wait for all force tasks to complete
    for (size_t b = 0; b < M; ++b)
        futures[b].get_void_result();
}

// ---------------------------------------------------------------------------
// Helper: compute energy via a single task
// ---------------------------------------------------------------------------
static double compute_energy(Runtime *runtime, Context ctx,
                             LogicalRegion q_lr, LogicalRegion p_lr)
{
    TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(NULL, 0));
    launcher.add_region_requirement(
        RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
    launcher.region_requirements[0].add_field(FID_VAL);
    launcher.add_region_requirement(
        RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
    launcher.region_requirements[1].add_field(FID_VAL);

    Future f = runtime->execute_task(ctx, launcher);
    return f.get_result<double>();
}

// ---------------------------------------------------------------------------
// Stepper coefficients for symplectic_rkn_sb3a_mclachlan (6 stages)
// ---------------------------------------------------------------------------
static const int NUM_STAGES = 6;

static void get_stepper_coefficients(double a[NUM_STAGES],
                                     double b[NUM_STAGES])
{
    const double w = 0.40518861839525227722;
    const double v = -0.28714404081652408900;
    a[0] = w;
    a[1] = v;
    a[2] = 0.5 - w - v;
    a[3] = a[2];
    a[4] = v;
    a[5] = w;

    b[0] = 0.0;
    b[1] = 1.0 / (2.0 - 4.0 * w);
    b[2] = 1.0 - 1.0 / (1.0 - 2.0 * w);
    b[3] = b[2];
    b[4] = b[1];
    b[5] = 0.0;
}

// ---------------------------------------------------------------------------
// TOP_LEVEL_TASK
// ---------------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // ---- command-line parsing ---------------------------------------------
    const InputArgs &cmd = Runtime::get_input_args();
    int    argc = cmd.argc;
    char **argv = cmd.argv;

    size_t N     = 1024;
    size_t G     = 128;
    size_t steps = 100;
    double dt    = 0.01;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc)
            N = static_cast<size_t>(std::stoull(argv[++i]));
        else if (std::strcmp(argv[i], "--G") == 0 && i + 1 < argc)
            G = static_cast<size_t>(std::stoull(argv[++i]));
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            steps = static_cast<size_t>(std::stoull(argv[++i]));
        else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
            dt = std::stod(argv[++i]);
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

    // ---- create index/field spaces and logical regions --------------------
    IndexSpace elem_is  = runtime->create_index_space(ctx,
                              Rect<1>(0, static_cast<coord_t>(N - 1)));
    IndexSpace block_is = runtime->create_index_space(ctx,
                              Rect<1>(0, static_cast<coord_t>(M - 1)));

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(double), FID_VAL);
    }

    LogicalRegion q_lr    = runtime->create_logical_region(ctx, elem_is, fs);
    LogicalRegion p_lr    = runtime->create_logical_region(ctx, elem_is, fs);
    LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, elem_is, fs);

    // ---- partitions -------------------------------------------------------
    IndexPartition block_ip = runtime->create_equal_partition(ctx, elem_is,
                                                              block_is);
    LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    block_ip);
    LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    block_ip);
    LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, block_ip);

    // ---- generate the initial momentum vector (same RNG as HPX code) ------
    dvec p_init(N);
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::mt19937 engine(0);
        auto gen = std::bind(dist, engine);
        std::generate(p_init.begin(), p_init.end(), std::ref(gen));
    }

    // ---- initialise q and p sub-regions -----------------------------------
    for (size_t b = 0; b < M; ++b) {
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(ctx, q_lp,
                                  static_cast<coord_t>(b));
        LogicalRegion p_sub = runtime->get_logical_subregion_by_color(ctx, p_lp,
                                  static_cast<coord_t>(b));

        // q <- 0
        {
            TaskLauncher launcher(INIT_Q_TASK_ID, TaskArgument(NULL, 0));
            launcher.add_region_requirement(
                RegionRequirement(q_sub, WRITE_DISCARD, EXCLUSIVE, q_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }

        // p <- p_init[b*G .. (b+1)*G)
        {
            InitPArgs hdr;
            hdr.offset = b * G;
            hdr.count  = G;
            size_t buf_sz = sizeof(InitPArgs) + N * sizeof(double);
            std::vector<char> buf(buf_sz);
            std::memcpy(buf.data(), &hdr, sizeof(hdr));
            std::memcpy(buf.data() + sizeof(hdr), p_init.data(),
                        N * sizeof(double));

            TaskLauncher launcher(INIT_P_TASK_ID,
                                  TaskArgument(buf.data(), buf_sz));
            launcher.add_region_requirement(
                RegionRequirement(p_sub, WRITE_DISCARD, EXCLUSIVE, p_lr));
            launcher.region_requirements[0].add_field(FID_VAL);
            runtime->execute_task(ctx, launcher);
        }
    }

    // Also initialise dpdt to zero
    runtime->fill_field(ctx, dpdt_lr, dpdt_lr, FID_VAL, 0.0);

    // ---- initial energy ---------------------------------------------------
    double e0 = compute_energy(runtime, ctx, q_lr, p_lr);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(e0)) << std::endl;

    // ---- stepper coefficients ---------------------------------------------
    double coef_a[NUM_STAGES], coef_b[NUM_STAGES];
    get_stepper_coefficients(coef_a, coef_b);

    // ---- integration loop (manual symplectic_rkn_sb3a_mclachlan) ----------
    for (size_t step = 0; step < steps; ++step) {
        for (int l = 0; l < NUM_STAGES; ++l) {
            // q drift (skip at l == 0, applied at the end instead)
            if (l > 0) {
                launch_scale_sum(runtime, ctx,
                                 q_lr, q_lp,
                                 p_lr, p_lp,
                                 block_is,
                                 1.0, coef_a[l] * dt);
            }

            // force evaluation
            launch_force(runtime, ctx,
                         q_lr, q_lp,
                         dpdt_lr, dpdt_lp,
                         block_is, M, G, N);

            // momentum kick
            if (coef_b[l] != 0.0) {
                launch_scale_sum(runtime, ctx,
                                 p_lr, p_lp,
                                 dpdt_lr, dpdt_lp,
                                 block_is,
                                 1.0, coef_b[l] * dt);
            }
        }
        // final drift with coef_a[0]
        launch_scale_sum(runtime, ctx,
                         q_lr, q_lp,
                         p_lr, p_lp,
                         block_is,
                         1.0, coef_a[0] * dt);
    }

    // ---- final energy -----------------------------------------------------
    double e1 = compute_energy(runtime, ctx, q_lr, p_lr);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(e1)) << std::endl;

    outfile.close();

    // ---- clean up ---------------------------------------------------------
    runtime->destroy_logical_region(ctx, q_lr);
    runtime->destroy_logical_region(ctx, p_lr);
    runtime->destroy_logical_region(ctx, dpdt_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, elem_is);
    runtime->destroy_index_space(ctx, block_is);
}

// ---------------------------------------------------------------------------
// main – register tasks and start the Legion runtime
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(INIT_Q_TASK_ID, "init_q");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<init_q_task>(registrar, "init_q");
    }
    {
        TaskVariantRegistrar registrar(INIT_P_TASK_ID, "init_p");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<init_p_task>(registrar, "init_p");
    }
    {
        TaskVariantRegistrar registrar(FORCE_TASK_ID, "force");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<force_task>(registrar, "force");
    }
    {
        TaskVariantRegistrar registrar(SCALE_SUM_TASK_ID, "scale_sum");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<scale_sum_task>(registrar, "scale_sum");
    }
    {
        TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
    }

    return Runtime::start(argc, argv);
}
