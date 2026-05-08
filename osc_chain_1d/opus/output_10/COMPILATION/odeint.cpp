// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion execution model
//
#include <legion.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <functional>
#include <cassert>

#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

// -----------------------------------------------------------
// Physics constants (from system.hpp)
// -----------------------------------------------------------
static const double KAPPA  = 3.5;
static const double LAMBDA = 4.5;

// -----------------------------------------------------------
// Task IDs
// -----------------------------------------------------------
enum {
  TOP_LEVEL_TASK_ID,
  COMPUTE_DPDT_TASK_ID,
  SCALE_SUM2_TASK_ID,
};

// -----------------------------------------------------------
// Field IDs
// -----------------------------------------------------------
enum {
  FID_VAL = 101,
};

// -----------------------------------------------------------
// Symplectic RKN SB3A McLachlan stepper coefficients (6 stages)
// (from boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp)
// -----------------------------------------------------------
static const int NUM_STAGES = 6;
static double coeff_a[NUM_STAGES];
static double coeff_b[NUM_STAGES];

static void init_coefficients() {
  coeff_a[0] =  0.40518861839525227722;
  coeff_a[1] = -0.28714404081652408900;
  coeff_a[2] =  0.5 - coeff_a[0] - coeff_a[1];
  coeff_a[3] =  coeff_a[2];
  coeff_a[4] =  coeff_a[1];
  coeff_a[5] =  coeff_a[0];

  coeff_b[0] = -3.0 / 73.0;
  coeff_b[1] =  17.0 / 59.0;
  coeff_b[2] =  1.0 - 2.0 * (coeff_b[0] + coeff_b[1]);
  coeff_b[3] =  coeff_b[1];
  coeff_b[4] =  coeff_b[0];
  coeff_b[5] =  0.0;
}

// -----------------------------------------------------------
// Math helpers (from system.hpp)
// -----------------------------------------------------------
namespace checked_math {
  inline double pow(double x, double y) {
    if (x == 0.0) return 0.0;
    return std::pow(std::abs(x), y);
  }
}

static inline double signed_pow(double x, double k) {
  return checked_math::pow(x, k) * boost::math::sign(x);
}

// -----------------------------------------------------------
// Task argument structures
// -----------------------------------------------------------
struct ComputeDpdtArgs {
  size_t block_index;
  size_t M; // total number of blocks
};

struct ScaleSum2Args {
  double alpha1;
  double alpha2;
};

// -----------------------------------------------------------
// compute_dpdt_task
//
// Computes the force (dp/dt) for one block of the oscillator chain.
//
// Region layout (variable number of regions):
//   Region 0 : q[block_idx]   – READ_ONLY
//   Region 1 : dpdt[block_idx] – WRITE_DISCARD
//   Region 2 : q[block_idx-1] – READ_ONLY  (present only if block_idx > 0)
//   Region 2 or 3 : q[block_idx+1] – READ_ONLY  (present only if block_idx < M-1)
// -----------------------------------------------------------
void compute_dpdt_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
  ComputeDpdtArgs args = *(const ComputeDpdtArgs *)task->args;
  const size_t bi = args.block_index;
  const size_t M  = args.M;
  const bool has_left  = (bi > 0);
  const bool has_right = (bi < M - 1);

  // --- access q block ---
  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
  Rect<1> q_rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());
  const size_t G = q_rect.hi[0] - q_rect.lo[0] + 1;
  const coord_t q_lo = q_rect.lo[0];

  // read q block into a local buffer for fast access
  std::vector<double> q(G);
  for (size_t i = 0; i < G; i++)
    q[i] = q_acc[q_lo + (coord_t)i];

  // --- boundary values from neighbours ---
  double q_l_val = 0.0, q_r_val = 0.0;
  int reg_idx = 2;
  if (has_left) {
    const FieldAccessor<READ_ONLY, double, 1> ql_acc(regions[reg_idx], FID_VAL);
    Rect<1> ql_rect = runtime->get_index_space_domain(ctx,
        task->regions[reg_idx].region.get_index_space());
    q_l_val = ql_acc[ql_rect.hi]; // last element of left neighbour
    reg_idx++;
  }
  if (has_right) {
    const FieldAccessor<READ_ONLY, double, 1> qr_acc(regions[reg_idx], FID_VAL);
    Rect<1> qr_rect = runtime->get_index_space_domain(ctx,
        task->regions[reg_idx].region.get_index_space());
    q_r_val = qr_acc[qr_rect.lo]; // first element of right neighbour
  }

  // --- compute forces (mirrors system_first/center/last_block) ---
  std::vector<double> dpdt(G);

  double coupling_lr;
  if (has_left)
    coupling_lr = -signed_pow(q[0] - q_l_val, LAMBDA - 1);
  else
    coupling_lr = -signed_pow(q[0], LAMBDA - 1);

  for (size_t i = 0; i < G - 1; i++) {
    dpdt[i]     = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
    coupling_lr =  signed_pow(q[i] - q[i + 1], LAMBDA - 1);
    dpdt[i]    -= coupling_lr;
  }

  if (has_right)
    dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr
                  - signed_pow(q[G - 1] - q_r_val, LAMBDA - 1);
  else
    dpdt[G - 1] = -signed_pow(q[G - 1], KAPPA - 1) + coupling_lr
                  - signed_pow(q[G - 1], LAMBDA - 1);

  // --- write dpdt ---
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);
  Rect<1> dpdt_rect = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  const coord_t dpdt_lo = dpdt_rect.lo[0];
  for (size_t i = 0; i < G; i++)
    dpdt_acc[dpdt_lo + (coord_t)i] = dpdt[i];
}

// -----------------------------------------------------------
// scale_sum2_task
//
// Performs:  x1[i] = alpha1 * x1[i] + alpha2 * x2[i]
//
// Region 0 : x1 – READ_WRITE  (destination, also first source)
// Region 1 : x2 – READ_ONLY   (second source)
//
// (mirrors shared_operations.hpp scale_sum2 with s1 == s2)
// -----------------------------------------------------------
void scale_sum2_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  ScaleSum2Args args = *(const ScaleSum2Args *)task->args;

  const FieldAccessor<READ_WRITE, double, 1> x1_acc(regions[0], FID_VAL);
  const FieldAccessor<READ_ONLY,  double, 1> x2_acc(regions[1], FID_VAL);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
      task->regions[0].region.get_index_space());

  for (PointInRectIterator<1> it(rect); it(); it++)
    x1_acc[*it] = args.alpha1 * x1_acc[*it] + args.alpha2 * x2_acc[*it];
}

// -----------------------------------------------------------
// Energy computation (sequential, via inline mapping)
// -----------------------------------------------------------
static double compute_energy(Context ctx, Runtime *runtime,
                             LogicalRegion q_lr, LogicalRegion p_lr,
                             size_t N)
{
  InlineLauncher q_il(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
  q_il.add_field(FID_VAL);
  PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
  q_pr.wait_until_valid();

  InlineLauncher p_il(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
  p_il.add_field(FID_VAL);
  PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
  p_pr.wait_until_valid();

  const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
  const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

  // Left boundary term
  double e = 0.5 * checked_math::pow(std::abs((double)q_acc[0]), LAMBDA) / LAMBDA;

  for (size_t i = 0; i < N - 1; ++i) {
    double qi  = q_acc[(coord_t)i];
    double pi  = p_acc[(coord_t)i];
    double qi1 = q_acc[(coord_t)(i + 1)];
    e += 0.5 * pi * pi
       + checked_math::pow(qi, KAPPA) / KAPPA
       + checked_math::pow(std::abs(qi - qi1), LAMBDA) / LAMBDA;
  }

  double qN = q_acc[(coord_t)(N - 1)];
  double pN = p_acc[(coord_t)(N - 1)];
  e += 0.5 * pN * pN
     + checked_math::pow(qN, KAPPA) / KAPPA
     + 0.5 * checked_math::pow(std::abs(qN), LAMBDA) / LAMBDA;

  runtime->unmap_region(ctx, q_pr);
  runtime->unmap_region(ctx, p_pr);
  return e;
}

// -----------------------------------------------------------
// Top-level task
// -----------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // ---- Initialise stepper coefficients ----
  init_coefficients();

  // ---- Parse command line ----
  size_t N     = 1024;
  size_t G     = 128;
  size_t steps = 100;
  double dt    = 0.01;

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

  assert(G > 0 && N % G == 0 && "N must be divisible by G");
  const size_t M = N / G;

  // ---- Output file ----
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

  // ---- Create index space [0, N-1] ----
  Rect<1> elem_rect(0, (coord_t)(N - 1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  runtime->attach_name(is, "element_is");

  // ---- Create field space ----
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
    alloc.allocate_field(sizeof(double), FID_VAL);
  }
  runtime->attach_name(fs, "val_fs");

  // ---- Create logical regions for q, p, dpdt ----
  LogicalRegion q_lr    = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion p_lr    = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion dpdt_lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(q_lr,    "q_lr");
  runtime->attach_name(p_lr,    "p_lr");
  runtime->attach_name(dpdt_lr, "dpdt_lr");

  // ---- Partition into M equal blocks ----
  Rect<1> color_rect(0, (coord_t)(M - 1));
  IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

  LogicalPartition q_lp    = runtime->get_logical_partition(ctx, q_lr,    ip);
  LogicalPartition p_lp    = runtime->get_logical_partition(ctx, p_lr,    ip);
  LogicalPartition dpdt_lp = runtime->get_logical_partition(ctx, dpdt_lr, ip);

  // ---- Initialise q = 0, dpdt = 0 ----
  runtime->fill_field<double>(ctx, q_lr,    q_lr,    FID_VAL, 0.0);
  runtime->fill_field<double>(ctx, dpdt_lr, dpdt_lr, FID_VAL, 0.0);

  // ---- Initialise p from deterministic PRNG (same as HPX version) ----
  {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);

    std::vector<double> p_init(N);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    InlineLauncher il(RegionRequirement(p_lr, WRITE_DISCARD, EXCLUSIVE, p_lr));
    il.add_field(FID_VAL);
    PhysicalRegion p_pr = runtime->map_region(ctx, il);
    p_pr.wait_until_valid();

    const FieldAccessor<WRITE_DISCARD, double, 1> p_acc(p_pr, FID_VAL);
    for (size_t i = 0; i < N; i++)
      p_acc[(coord_t)i] = p_init[i];

    runtime->unmap_region(ctx, p_pr);
  }

  // ---- Compute initial energy ----
  {
    double e = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(e)) << std::endl;
  }

  // ============================================================
  // Time integration loop
  //
  // Implements symplectic_rkn_sb3a_mclachlan::do_step manually.
  // Following boost's symplectic_nystroem_stepper_base::do_step_impl,
  // for each stage l in [0..5]:
  //   q    = 1.0 * q + a[l]*dt * p       (position drift)
  //   dpdt = f(q)                         (system evaluation)
  //   p    = 1.0 * p + b[l]*dt * dpdt     (momentum kick)
  //
  // Legion's automatic dependence analysis on the logical
  // (sub-)regions ensures correct task ordering without
  // explicit barriers.
  // ============================================================
  for (size_t step = 0; step < steps; step++) {
    for (int stage = 0; stage < NUM_STAGES; stage++) {

      // ------ 1. q = 1.0*q + a[stage]*dt * p (position drift) ------
      {
        ScaleSum2Args ss_args;
        ss_args.alpha1 = 1.0;
        ss_args.alpha2 = coeff_a[stage] * dt;

        IndexTaskLauncher launcher(SCALE_SUM2_TASK_ID, color_is,
                               TaskArgument(&ss_args, sizeof(ss_args)),
                               ArgumentMap());
        // Region 0: q partition – READ_WRITE
        launcher.add_region_requirement(
            RegionRequirement(q_lp, 0 /*identity proj*/,
                              READ_WRITE, EXCLUSIVE, q_lr));
        launcher.add_field(0, FID_VAL);
        // Region 1: p partition – READ_ONLY
        launcher.add_region_requirement(
            RegionRequirement(p_lp, 0 /*identity proj*/,
                              READ_ONLY, EXCLUSIVE, p_lr));
        launcher.add_field(1, FID_VAL);

        runtime->execute_index_space(ctx, launcher);
      }

      // ------ 2. dpdt = f(q) ------
      // Launch M individual tasks (variable region requirements
      // due to neighbour access).
      for (size_t i = 0; i < M; i++) {
        ComputeDpdtArgs dpdt_args;
        dpdt_args.block_index = i;
        dpdt_args.M           = M;

        TaskLauncher launcher(COMPUTE_DPDT_TASK_ID,
                              TaskArgument(&dpdt_args, sizeof(dpdt_args)));

        // Region 0: q[i] READ_ONLY
        LogicalRegion q_sub = runtime->get_logical_subregion_by_color(ctx,
            q_lp, DomainPoint(Point<1>((coord_t)i)));
        launcher.add_region_requirement(
            RegionRequirement(q_sub, READ_ONLY, EXCLUSIVE, q_lr));
        launcher.add_field(0, FID_VAL);

        // Region 1: dpdt[i] WRITE_DISCARD
        LogicalRegion dpdt_sub = runtime->get_logical_subregion_by_color(ctx,
            dpdt_lp, DomainPoint(Point<1>((coord_t)i)));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub, WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
        launcher.add_field(1, FID_VAL);

        int reg_idx = 2;

        // Region 2 (optional): q[i-1] READ_ONLY
        if (i > 0) {
          LogicalRegion q_left = runtime->get_logical_subregion_by_color(ctx,
              q_lp, DomainPoint(Point<1>((coord_t)(i - 1))));
          launcher.add_region_requirement(
              RegionRequirement(q_left, READ_ONLY, EXCLUSIVE, q_lr));
          launcher.add_field(reg_idx, FID_VAL);
          reg_idx++;
        }

        // Region 2 or 3 (optional): q[i+1] READ_ONLY
        if (i < M - 1) {
          LogicalRegion q_right = runtime->get_logical_subregion_by_color(ctx,
              q_lp, DomainPoint(Point<1>((coord_t)(i + 1))));
          launcher.add_region_requirement(
              RegionRequirement(q_right, READ_ONLY, EXCLUSIVE, q_lr));
          launcher.add_field(reg_idx, FID_VAL);
        }

        runtime->execute_task(ctx, launcher);
      }

      // ------ 3. p = 1.0*p + b[stage]*dt * dpdt (momentum kick) ------
      {
        ScaleSum2Args ss_args;
        ss_args.alpha1 = 1.0;
        ss_args.alpha2 = coeff_b[stage] * dt;

        IndexTaskLauncher launcher(SCALE_SUM2_TASK_ID, color_is,
                               TaskArgument(&ss_args, sizeof(ss_args)),
                               ArgumentMap());
        // Region 0: p partition – READ_WRITE
        launcher.add_region_requirement(
            RegionRequirement(p_lp, 0 /*identity proj*/,
                              READ_WRITE, EXCLUSIVE, p_lr));
        launcher.add_field(0, FID_VAL);
        // Region 1: dpdt partition – READ_ONLY
        launcher.add_region_requirement(
            RegionRequirement(dpdt_lp, 0 /*identity proj*/,
                              READ_ONLY, EXCLUSIVE, dpdt_lr));
        launcher.add_field(1, FID_VAL);

        runtime->execute_index_space(ctx, launcher);
      }

    } // stage
  } // step

  // ---- Compute final energy ----
  {
    double e = compute_energy(ctx, runtime, q_lr, p_lr, N);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(e)) << std::endl;
  }

  outfile.close();

  // ---- Cleanup ----
  runtime->destroy_logical_region(ctx, dpdt_lr);
  runtime->destroy_logical_region(ctx, p_lr);
  runtime->destroy_logical_region(ctx, q_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
}

// -----------------------------------------------------------
// main – register tasks and start the Legion runtime
// -----------------------------------------------------------
int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  // Top-level task
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // compute_dpdt (individual task launches)
  {
    TaskVariantRegistrar registrar(COMPUTE_DPDT_TASK_ID, "compute_dpdt");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<compute_dpdt_task>(registrar, "compute_dpdt");
  }

  // scale_sum2 (index task launches)
  {
    TaskVariantRegistrar registrar(SCALE_SUM2_TASK_ID, "scale_sum2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<scale_sum2_task>(registrar, "scale_sum2");
  }

  return Runtime::start(argc, argv);
}
