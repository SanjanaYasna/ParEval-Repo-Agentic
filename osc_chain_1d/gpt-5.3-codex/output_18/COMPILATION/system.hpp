// Copyright 2013 Mario Mulansky
// Translated from HPX to Legion

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include "legion.h"

#include <vector>
#include <cmath>
#include <cassert>

using namespace Legion;

constexpr double KAPPA  = 3.5;
constexpr double LAMBDA = 4.5;

// Keep field/task ids local to this header translation unit.
enum SystemFieldIDs : FieldID {
  FID_VAL = 0
};

enum SystemTaskIDs : TaskID {
  OSC_CHAIN_BLOCK_TASK_ID = 1001
};

namespace checked_math {
inline double pow(double x, double y) {
  if (x == 0.0) return 0.0;
  using std::abs;
  using std::pow;
  return pow(abs(x), y);
}
}  // namespace checked_math

inline double signed_pow(double x, double k) {
  const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  return checked_math::pow(x, k) * s;
}

struct OscChainTaskArgs {
  coord_t q_lo;
  coord_t q_hi;
};

// One Legion task computes dpdt for one block (one subregion of dpdt).
inline void osc_chain_block_task(const Task* task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx,
                                 Runtime* runtime) {
  assert(task->arglen == sizeof(OscChainTaskArgs));
  const auto* args = static_cast<const OscChainTaskArgs*>(task->args);

  assert(regions.size() == 2);

  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

  const Domain dpdt_dom =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());
  const Rect<1> r = dpdt_dom.get_rect<1>();

  const coord_t lo = r.lo[0];
  const coord_t hi = r.hi[0];
  if (lo > hi) return;

  const double q_l = (lo == args->q_lo) ? 0.0 : q_acc[Point<1>(lo - 1)];
  const double q_r = (hi == args->q_hi) ? 0.0 : q_acc[Point<1>(hi + 1)];

  double coupling_lr =
      -signed_pow(q_acc[Point<1>(lo)] - q_l, LAMBDA - 1.0);

  for (coord_t i = lo; i < hi; ++i) {
    const double qi   = q_acc[Point<1>(i)];
    const double qip1 = q_acc[Point<1>(i + 1)];

    dpdt_acc[Point<1>(i)] = -signed_pow(qi, KAPPA - 1.0) + coupling_lr;
    coupling_lr = signed_pow(qi - qip1, LAMBDA - 1.0);
    dpdt_acc[Point<1>(i)] -= coupling_lr;
  }

  const double qhi = q_acc[Point<1>(hi)];
  dpdt_acc[Point<1>(hi)] =
      -signed_pow(qhi, KAPPA - 1.0) + coupling_lr
      - signed_pow(qhi - q_r, LAMBDA - 1.0);
}

// Asynchronous launch (HPX osc_chain equivalent without global barrier).
inline FutureMap osc_chain(Runtime* runtime,
                           Context ctx,
                           LogicalRegion q_lr,
                           LogicalRegion dpdt_lr,
                           LogicalPartition dpdt_lp) {
  const Domain q_dom =
      runtime->get_index_space_domain(ctx, q_lr.get_index_space());
  const Rect<1> q_rect = q_dom.get_rect<1>();

  const OscChainTaskArgs args{q_rect.lo[0], q_rect.hi[0]};

  const Domain color_space =
      runtime->get_index_partition_color_space(ctx, dpdt_lp.get_index_partition());

  ArgumentMap arg_map;
  IndexLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, color_space,
                         TaskArgument(&args, sizeof(args)), arg_map);

  launcher.add_region_requirement(
      RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
  launcher.region_requirements.back().add_field(FID_VAL);

  launcher.add_region_requirement(
      RegionRequirement(dpdt_lp, 0 /* identity projection */,
                        WRITE_DISCARD, EXCLUSIVE, dpdt_lr));
  launcher.region_requirements.back().add_field(FID_VAL);

  return runtime->execute_index_space(ctx, launcher);
}

// Synchronous launch (HPX osc_chain_gb equivalent with global barrier).
inline void osc_chain_gb(Runtime* runtime,
                         Context ctx,
                         LogicalRegion q_lr,
                         LogicalRegion dpdt_lr,
                         LogicalPartition dpdt_lp) {
  FutureMap fm = osc_chain(runtime, ctx, q_lr, dpdt_lr, dpdt_lp);
  fm.wait_all_results();
  runtime->issue_execution_fence(ctx);
}

inline double energy(const std::vector<double>& q, const std::vector<double>& p) {
  assert(q.size() == p.size());
  if (q.empty()) return 0.0;

  using checked_math::pow;
  using std::abs;

  const size_t N = q.size();
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

// Region-based energy helper (maps q and p regions and computes scalar energy).
inline double energy(Runtime* runtime,
                     Context ctx,
                     LogicalRegion q_lr,
                     LogicalRegion p_lr) {
  InlineLauncher q_il(RegionRequirement(q_lr, READ_ONLY, EXCLUSIVE, q_lr));
  q_il.requirement.add_field(FID_VAL);
  PhysicalRegion q_pr = runtime->map_region(ctx, q_il);
  q_pr.wait_until_valid();

  InlineLauncher p_il(RegionRequirement(p_lr, READ_ONLY, EXCLUSIVE, p_lr));
  p_il.requirement.add_field(FID_VAL);
  PhysicalRegion p_pr = runtime->map_region(ctx, p_il);
  p_pr.wait_until_valid();

  const FieldAccessor<READ_ONLY, double, 1> q_acc(q_pr, FID_VAL);
  const FieldAccessor<READ_ONLY, double, 1> p_acc(p_pr, FID_VAL);

  const Rect<1> q_rect =
      runtime->get_index_space_domain(ctx, q_lr.get_index_space()).get_rect<1>();
  const Rect<1> p_rect =
      runtime->get_index_space_domain(ctx, p_lr.get_index_space()).get_rect<1>();

  assert((q_rect.hi[0] - q_rect.lo[0]) == (p_rect.hi[0] - p_rect.lo[0]));

  using checked_math::pow;
  using std::abs;

  const coord_t lo = q_rect.lo[0];
  const coord_t hi = q_rect.hi[0];

  double e = 0.0;
  if (lo <= hi) {
    e = 0.5 * pow(abs(q_acc[Point<1>(lo)]), LAMBDA) / LAMBDA;

    for (coord_t i = lo; i < hi; ++i) {
      const double qi   = q_acc[Point<1>(i)];
      const double qip1 = q_acc[Point<1>(i + 1)];
      const double pi   = p_acc[Point<1>(i)];
      e += 0.5 * pi * pi
         + pow(qi, KAPPA) / KAPPA
         + pow(abs(qi - qip1), LAMBDA) / LAMBDA;
    }

    const double qn = q_acc[Point<1>(hi)];
    const double pn = p_acc[Point<1>(hi)];
    e += 0.5 * pn * pn
       + pow(qn, KAPPA) / KAPPA
       + 0.5 * pow(abs(qn), LAMBDA) / LAMBDA;
  }

  runtime->unmap_region(ctx, q_pr);
  runtime->unmap_region(ctx, p_pr);
  return e;
}

inline void preregister_system_tasks() {
  TaskVariantRegistrar registrar(OSC_CHAIN_BLOCK_TASK_ID, "osc_chain_block_task");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<osc_chain_block_task>(registrar, "osc_chain_block_task");
}

#endif  // SYSTEM_HPP
