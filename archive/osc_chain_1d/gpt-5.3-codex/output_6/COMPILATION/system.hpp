// Copyright 2013 Mario Mulansky
// Translated to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <boost/math/special_functions/sign.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y) {
  if (x == 0.0) return 0.0;
  using std::abs;
  using std::pow;
  return pow(abs(x), y);
}
}  // namespace checked_math

inline double signed_pow(double x, double k) {
  using boost::math::sign;
  return checked_math::pow(x, k) * sign(x);
}

typedef std::vector<double> dvec;

// Single scalar field per element (q or dpdt/p)
enum FieldIDs : Legion::FieldID { FID_VALUE = 0 };

// Task IDs for this header
enum TaskIDs : Legion::TaskID { SYSTEM_BLOCK_TASK_ID = 10010 };

// Legion-backed state handle (logical region + partition into blocks)
struct state_type {
  Legion::Runtime* runtime{nullptr};
  Legion::Context context{};
  Legion::LogicalRegion lr{};
  Legion::LogicalPartition lp{};  // partition of lr into blocks
  Legion::IndexSpace block_is{};  // color space for block index launch
};

struct SystemTaskArgs {
  std::int64_t lo;
  std::int64_t hi;
};

inline void system_block_task(const Legion::Task* task,
                              const std::vector<Legion::PhysicalRegion>& regions,
                              Legion::Context ctx,
                              Legion::Runtime* runtime) {
  assert(task != nullptr);
  assert(task->args != nullptr);
  assert(regions.size() == 2);

  const auto* args = reinterpret_cast<const SystemTaskArgs*>(task->args);
  const Legion::coord_t global_lo = static_cast<Legion::coord_t>(args->lo);
  const Legion::coord_t global_hi = static_cast<Legion::coord_t>(args->hi);

  const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(regions[0], FID_VALUE);
  Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VALUE);

  const Legion::Rect<1> out_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  for (Legion::PointInRectIterator<1> pir(out_rect); pir(); pir++) {
    const Legion::coord_t idx = (*pir)[0];

    const double q_i = q_acc[*pir];
    const double q_l =
        (idx == global_lo) ? 0.0 : q_acc[Legion::Point<1>(idx - 1)];
    const double q_r =
        (idx == global_hi) ? 0.0 : q_acc[Legion::Point<1>(idx + 1)];

    // Equivalent to the first/center/last block HPX implementations
    const double val =
        -signed_pow(q_i, KAPPA - 1.0) -
        signed_pow(q_i - q_l, LAMBDA - 1.0) -
        signed_pow(q_i - q_r, LAMBDA - 1.0);

    dpdt_acc[*pir] = val;
  }
}

inline void register_system_tasks() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  Legion::TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block_task");
  registrar.add_constraint(
      Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
  Legion::Runtime::preregister_task_variant<system_block_task>(
      registrar, "system_block_task");
}

inline Legion::FutureMap launch_osc_chain(state_type& q, state_type& dpdt) {
  assert(q.runtime != nullptr);
  assert(dpdt.runtime == q.runtime);

  const Legion::Rect<1> q_rect =
      q.runtime->get_index_space_domain(q.context, q.lr.get_index_space());

  SystemTaskArgs args{
      static_cast<std::int64_t>(q_rect.lo[0]),
      static_cast<std::int64_t>(q_rect.hi[0]),
  };

  Legion::IndexLauncher launcher(
      SYSTEM_BLOCK_TASK_ID,
      dpdt.block_is,
      Legion::TaskArgument(&args, sizeof(args)),
      Legion::ArgumentMap());

  // q: read-only full region (neighbors are read from adjacent global points)
  launcher.add_region_requirement(
      Legion::RegionRequirement(q.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, q.lr));
  launcher.region_requirements.back().add_field(FID_VALUE);

  // dpdt: write-discard on per-block partition
  launcher.add_region_requirement(
      Legion::RegionRequirement(dpdt.lp, 0 /* identity projection */,
                                Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt.lr));
  launcher.region_requirements.back().add_field(FID_VALUE);

  return q.runtime->execute_index_space(q.context, launcher);
}

// Asynchronous launch (no explicit global barrier)
inline void osc_chain(state_type& q, state_type& dpdt) {
  (void)launch_osc_chain(q, dpdt);
}

// Launch + explicit global barrier (wait for all block tasks)
inline void osc_chain_gb(state_type& q, state_type& dpdt) {
  Legion::FutureMap fm = launch_osc_chain(q, dpdt);
  fm.wait_all_results();
}

inline double energy(const dvec& q, const dvec& p) {
  using std::abs;
  const size_t N = q.size();

  double e = 0.5 * checked_math::pow(abs(q[0]), LAMBDA) / LAMBDA;
  for (size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i] + checked_math::pow(q[i], KAPPA) / KAPPA +
         checked_math::pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1] +
       checked_math::pow(q[N - 1], KAPPA) / KAPPA +
       0.5 * checked_math::pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
  return e;
}

inline double energy(const state_type& q_state, const state_type& p_state) {
  assert(q_state.runtime != nullptr);
  assert(q_state.runtime == p_state.runtime);

  Legion::InlineLauncher q_inline(
      Legion::RegionRequirement(q_state.lr, Legion::READ_ONLY,
                                Legion::EXCLUSIVE, q_state.lr));
  q_inline.requirement.add_field(FID_VALUE);

  Legion::InlineLauncher p_inline(
      Legion::RegionRequirement(p_state.lr, Legion::READ_ONLY,
                                Legion::EXCLUSIVE, p_state.lr));
  p_inline.requirement.add_field(FID_VALUE);

  Legion::PhysicalRegion q_pr =
      q_state.runtime->map_region(q_state.context, q_inline);
  Legion::PhysicalRegion p_pr =
      p_state.runtime->map_region(p_state.context, p_inline);

  q_pr.wait_until_valid();
  p_pr.wait_until_valid();

  const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(q_pr, FID_VALUE);
  const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> p_acc(p_pr, FID_VALUE);

  const Legion::Rect<1> q_rect =
      q_state.runtime->get_index_space_domain(q_state.context, q_state.lr.get_index_space());
  const Legion::Rect<1> p_rect =
      p_state.runtime->get_index_space_domain(p_state.context, p_state.lr.get_index_space());

  dvec q_vals;
  dvec p_vals;
  q_vals.reserve(static_cast<size_t>(q_rect.volume()));
  p_vals.reserve(static_cast<size_t>(p_rect.volume()));

  for (Legion::PointInRectIterator<1> pir(q_rect); pir(); pir++) {
    q_vals.push_back(q_acc[*pir]);
  }
  for (Legion::PointInRectIterator<1> pir(p_rect); pir(); pir++) {
    p_vals.push_back(p_acc[*pir]);
  }

  q_state.runtime->unmap_region(q_state.context, q_pr);
  p_state.runtime->unmap_region(p_state.context, p_pr);

  return energy(q_vals, p_vals);
}

#endif
