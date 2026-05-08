// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion tasks

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <cassert>
#include <cmath>
#include <cstddef>
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
  const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  return checked_math::pow(x, k) * s;
}

typedef std::vector<double> dvec;

// -----------------------------------------------------------------------------
// Legion state representation (1D field partitioned into equal-sized blocks)
// -----------------------------------------------------------------------------
struct state_type {
  Legion::Runtime* runtime{nullptr};
  Legion::Context ctx{};

  Legion::IndexSpace is;
  Legion::FieldSpace fs;
  Legion::LogicalRegion lr;

  Legion::IndexSpace color_is;
  Legion::IndexPartition ip_blocks;
  Legion::LogicalPartition lp_blocks;

  Legion::FieldID fid{1};

  std::size_t N{0};  // global size
  std::size_t G{0};  // block size
  std::size_t M{0};  // number of blocks
};

enum : Legion::TaskID {
  OSC_CHAIN_BLOCK_TASK_ID = 1001
};

struct osc_chain_task_args {
  Legion::coord_t N;
};

inline state_type make_state(Legion::Runtime* runtime,
                             Legion::Context ctx,
                             std::size_t N,
                             std::size_t G,
                             Legion::FieldID fid = 1) {
  assert(runtime != nullptr);
  assert(G > 0);
  assert((N % G) == 0 && "N must be divisible by G");

  state_type s;
  s.runtime = runtime;
  s.ctx = ctx;
  s.N = N;
  s.G = G;
  s.M = N / G;
  s.fid = fid;

  s.is = runtime->create_index_space(
      ctx, Legion::Rect<1>(0, static_cast<Legion::coord_t>(N - 1)));

  s.fs = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator alloc = runtime->create_field_allocator(ctx, s.fs);
    alloc.allocate_field(sizeof(double), s.fid);
  }

  s.lr = runtime->create_logical_region(ctx, s.is, s.fs);

  s.color_is = runtime->create_index_space(
      ctx, Legion::Rect<1>(0, static_cast<Legion::coord_t>(s.M - 1)));

  Legion::Transform<1, 1> transform;
  transform[0][0] = static_cast<Legion::coord_t>(G);
  Legion::Rect<1> extent(0, static_cast<Legion::coord_t>(G - 1));

  s.ip_blocks = runtime->create_partition_by_restriction(
      ctx, s.is, s.color_is, transform, extent);

  s.lp_blocks = runtime->get_logical_partition(ctx, s.lr, s.ip_blocks);

  return s;
}

inline void destroy_state(state_type& s) {
  if (!s.runtime) return;
  s.runtime->destroy_logical_region(s.ctx, s.lr);
  s.runtime->destroy_field_space(s.ctx, s.fs);
  s.runtime->destroy_index_partition(s.ctx, s.ip_blocks);
  s.runtime->destroy_index_space(s.ctx, s.color_is);
  s.runtime->destroy_index_space(s.ctx, s.is);
  s = state_type{};
}

// -----------------------------------------------------------------------------
// Legion task: compute dpdt for one block
// -----------------------------------------------------------------------------
inline void osc_chain_block_task(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& regions,
                                 Legion::Context ctx,
                                 Legion::Runtime* runtime) {
  assert(task->arglen == sizeof(osc_chain_task_args));
  const auto* args = static_cast<const osc_chain_task_args*>(task->args);
  const Legion::coord_t N = args->N;

  // regions[0] = full q (READ_ONLY)
  // regions[1] = one dpdt block (WRITE_DISCARD)
  Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(regions[0], task->regions[0].instance_fields[0]);
  Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> dpdt_acc(regions[1], task->regions[1].instance_fields[0]);

  const Legion::Rect<1> block_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  for (Legion::coord_t i = block_rect.lo[0]; i <= block_rect.hi[0]; ++i) {
    const double qi = q_acc[Legion::Point<1>(i)];

    const double onsite = -signed_pow(qi, KAPPA - 1.0);

    const double left_coupling =
        (i == 0) ? -signed_pow(qi, LAMBDA - 1.0)
                 : signed_pow(q_acc[Legion::Point<1>(i - 1)] - qi, LAMBDA - 1.0);

    const double right_flux =
        (i == (N - 1)) ? signed_pow(qi, LAMBDA - 1.0)
                       : signed_pow(qi - q_acc[Legion::Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[Legion::Point<1>(i)] = onsite + left_coupling - right_flux;
  }
}

inline void preregister_system_tasks() {
  Legion::TaskVariantRegistrar registrar(OSC_CHAIN_BLOCK_TASK_ID, "osc_chain_block_task");
  registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
  Legion::Runtime::preregister_task_variant<osc_chain_block_task>(registrar, "osc_chain_block_task");
}

// -----------------------------------------------------------------------------
// System evaluation (Legion equivalent of HPX dataflow graph)
// -----------------------------------------------------------------------------
inline void osc_chain(state_type& q, state_type& dpdt) {
  assert(q.runtime == dpdt.runtime);
  assert(q.ctx == dpdt.ctx);
  assert(q.N == dpdt.N);
  assert(q.M == dpdt.M);

  osc_chain_task_args args{static_cast<Legion::coord_t>(q.N)};
  Legion::IndexLauncher launcher(
      OSC_CHAIN_BLOCK_TASK_ID,
      Legion::Rect<1>(0, static_cast<Legion::coord_t>(q.M - 1)),
      Legion::TaskArgument(&args, sizeof(args)),
      Legion::ArgumentMap());

  launcher.add_region_requirement(
      Legion::RegionRequirement(q.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, q.lr));
  launcher.region_requirements.back().add_field(q.fid);

  launcher.add_region_requirement(
      Legion::RegionRequirement(dpdt.lp_blocks, 0, Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt.lr));
  launcher.region_requirements.back().add_field(dpdt.fid);

  (void)q.runtime->execute_index_space(q.ctx, launcher);  // async launch
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
  assert(q.runtime == dpdt.runtime);
  assert(q.ctx == dpdt.ctx);
  assert(q.N == dpdt.N);
  assert(q.M == dpdt.M);

  osc_chain_task_args args{static_cast<Legion::coord_t>(q.N)};
  Legion::IndexLauncher launcher(
      OSC_CHAIN_BLOCK_TASK_ID,
      Legion::Rect<1>(0, static_cast<Legion::coord_t>(q.M - 1)),
      Legion::TaskArgument(&args, sizeof(args)),
      Legion::ArgumentMap());

  launcher.add_region_requirement(
      Legion::RegionRequirement(q.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, q.lr));
  launcher.region_requirements.back().add_field(q.fid);

  launcher.add_region_requirement(
      Legion::RegionRequirement(dpdt.lp_blocks, 0, Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt.lr));
  launcher.region_requirements.back().add_field(dpdt.fid);

  Legion::FutureMap fm = q.runtime->execute_index_space(q.ctx, launcher);
  fm.wait_all_results();  // global barrier
}

// -----------------------------------------------------------------------------
// Energy
// -----------------------------------------------------------------------------
inline double energy(const dvec& q, const dvec& p) {
  using checked_math::pow;
  using std::abs;
  const std::size_t N = q.size();

  double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
       + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
  return e;
}

inline dvec materialize_state(const state_type& s) {
  Legion::InlineLauncher launcher(
      Legion::RegionRequirement(s.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, s.lr));
  launcher.requirement.add_field(s.fid);

  Legion::PhysicalRegion pr = s.runtime->map_region(s.ctx, launcher);
  pr.wait_until_valid();

  Legion::FieldAccessor<Legion::READ_ONLY, double, 1> acc(pr, s.fid);
  const Legion::Rect<1> rect =
      s.runtime->get_index_space_domain(s.ctx, s.is);

  dvec out;
  out.reserve(s.N);
  for (Legion::coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    out.push_back(acc[Legion::Point<1>(i)]);
  }

  s.runtime->unmap_region(s.ctx, pr);
  return out;
}

inline double energy(const state_type& q_state, const state_type& p_state) {
  dvec q = materialize_state(q_state);
  dvec p = materialize_state(p_state);
  return energy(q, p);
}

#endif
