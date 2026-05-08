// Translated from HPX to Legion execution model
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>

using namespace Legion;

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

//------------------------------------------------------------------------------
// Math helpers
//------------------------------------------------------------------------------
namespace checked_math {
inline double pow(double x, double y) {
  if (x == 0.0) return 0.0;
  using std::abs;
  using std::pow;
  return pow(abs(x), y);
}
} // namespace checked_math

inline double signum(double x) {
  return (x > 0.0) - (x < 0.0);
}

inline double signed_pow(double x, double k) {
  return checked_math::pow(x, k) * signum(x);
}

//------------------------------------------------------------------------------
// Legion IDs used by this file
//------------------------------------------------------------------------------
enum SystemTaskIDs : TaskID {
  SYSTEM_BLOCK_TASK_ID = 1000,
  ENERGY_TASK_ID       = 1001
};

enum SystemFieldIDs : FieldID {
  FID_X = 2000   // field that stores scalar values (q, p, dpdt)
};

//------------------------------------------------------------------------------
// Local vector energy helper (kept from original semantics)
//------------------------------------------------------------------------------
using dvec = std::vector<double>;

inline double energy(const dvec &q, const dvec &p) {
  using checked_math::pow;
  using std::abs;

  const size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
  for (size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i]
       + pow(q[i], KAPPA) / KAPPA
       + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
  return e;
}

//------------------------------------------------------------------------------
// Task: compute dpdt for one block (index-space point)
// regions[0]: q     (READ_ONLY, full region)
// regions[1]: dpdt  (WRITE_DISCARD, projected subregion for this block)
//------------------------------------------------------------------------------
inline void system_block_task(const Task *task,
                              const std::vector<PhysicalRegion> &regions,
                              Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_X);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_X);

  const Rect<1> q_rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  const Rect<1> blk_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  const int64_t g_lo = q_rect.lo[0];
  const int64_t g_hi = q_rect.hi[0];

  for (int64_t i = blk_rect.lo[0]; i <= blk_rect.hi[0]; ++i) {
    const double qi   = q_acc[Point<1>(i)];
    const double qim1 = (i == g_lo) ? 0.0 : q_acc[Point<1>(i - 1)];
    const double qip1 = (i == g_hi) ? 0.0 : q_acc[Point<1>(i + 1)];

    // Equivalent to original first/center/last block logic:
    // dpdt_i = -|q_i|^(KAPPA-1) sign(q_i)
    //        + |q_{i-1}-q_i|^(LAMBDA-1) sign(q_{i-1}-q_i)
    //        - |q_i-q_{i+1}|^(LAMBDA-1) sign(q_i-q_{i+1})
    const double onsite = -signed_pow(qi, KAPPA - 1.0);
    const double left_c =  signed_pow(qim1 - qi, LAMBDA - 1.0);
    const double right_c = -signed_pow(qi - qip1, LAMBDA - 1.0);

    dpdt_acc[Point<1>(i)] = onsite + left_c + right_c;
  }
}

//------------------------------------------------------------------------------
// Launch helpers for osc_chain
//
// q_lr    : full q region (FID_X)
// dpdt_lr : full dpdt region (FID_X)
// dpdt_lp : disjoint partition of dpdt_lr into blocks (colors = tasks)
//------------------------------------------------------------------------------
inline FutureMap launch_osc_chain(Context ctx, Runtime *runtime,
                                  LogicalRegion q_lr,
                                  LogicalRegion dpdt_lr,
                                  LogicalPartition dpdt_lp) {
  IndexSpace color_is =
      runtime->get_index_partition_color_space(ctx, dpdt_lp.get_index_partition());
  Domain color_dom = runtime->get_index_space_domain(ctx, color_is);

  IndexLauncher launcher(SYSTEM_BLOCK_TASK_ID, color_dom, TaskArgument(nullptr, 0), ArgumentMap());

  // Requirement 0: every task reads full q
  launcher.add_region_requirement(
      RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
  launcher.add_field(0, FID_X);

  // Requirement 1: every task writes only its own dpdt block (projection 0 = identity)
  launcher.add_region_requirement(
      RegionRequirement(dpdt_lp, 0 /*identity projection*/,
                        WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
  launcher.add_field(1, FID_X);

  return runtime->execute_index_space(ctx, launcher);
}

// async version (no global barrier)
inline void osc_chain(Context ctx, Runtime *runtime,
                      LogicalRegion q_lr,
                      LogicalRegion dpdt_lr,
                      LogicalPartition dpdt_lp) {
  (void)launch_osc_chain(ctx, runtime, q_lr, dpdt_lr, dpdt_lp);
}

// global-barrier version (wait for all blocks)
inline void osc_chain_gb(Context ctx, Runtime *runtime,
                         LogicalRegion q_lr,
                         LogicalRegion dpdt_lr,
                         LogicalPartition dpdt_lp) {
  FutureMap fm = launch_osc_chain(ctx, runtime, q_lr, dpdt_lr, dpdt_lp);
  fm.wait_all_results();
}

//------------------------------------------------------------------------------
// Energy task over full regions
// regions[0]: q (READ_ONLY)
// regions[1]: p (READ_ONLY)
//------------------------------------------------------------------------------
inline double energy_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_X);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_X);

  const Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  if (rect.empty()) return 0.0;

  using checked_math::pow;
  using std::abs;

  const int64_t lo = rect.lo[0];
  const int64_t hi = rect.hi[0];

  double e = 0.5 * pow(abs(q_acc[Point<1>(lo)]), LAMBDA) / LAMBDA;
  for (int64_t i = lo; i < hi; ++i) {
    const double qi  = q_acc[Point<1>(i)];
    const double qip = q_acc[Point<1>(i + 1)];
    const double pi  = p_acc[Point<1>(i)];
    e += 0.5 * pi * pi
       + pow(qi, KAPPA) / KAPPA
       + pow(abs(qi - qip), LAMBDA) / LAMBDA;
  }

  const double qn = q_acc[Point<1>(hi)];
  const double pn = p_acc[Point<1>(hi)];
  e += 0.5 * pn * pn
     + pow(qn, KAPPA) / KAPPA
     + 0.5 * pow(abs(qn), LAMBDA) / LAMBDA;

  return e;
}

inline double energy(Context ctx, Runtime *runtime,
                     LogicalRegion q_lr, LogicalRegion p_lr) {
  TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(nullptr, 0));

  launcher.add_region_requirement(
      RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
  launcher.add_field(0, FID_X);

  launcher.add_region_requirement(
      RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
  launcher.add_field(1, FID_X);

  Future f = runtime->execute_task(ctx, launcher);
  return f.get_result<double>();
}

//------------------------------------------------------------------------------
// Call once during startup (before Runtime::start)
//------------------------------------------------------------------------------
inline void preregister_system_tasks() {
  {
    TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<system_block_task>(registrar, "system_block_task");
  }
  {
    TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<double, energy_task>(registrar, "energy_task");
  }
}

#endif // SYSTEM_HPP
