// Copyright 2013 Mario Mulansky
// Translated to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cassert>

#include <legion.h>

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

using dvec = std::vector<double>;

namespace legion_osc_chain {

using namespace Legion;

enum SystemTaskIDs : TaskID {
  OSC_CHAIN_BLOCK_TASK_ID = 10001
};

enum SystemFieldIDs : FieldID {
  FID_VALUE = 100
};

struct state_type {
  Runtime* runtime{nullptr};
  Context ctx{};

  std::size_t N{0};  // global vector length
  std::size_t G{0};  // block size
  std::size_t M{0};  // number of blocks

  IndexSpace is;
  FieldSpace fs;
  LogicalRegion lr;

  IndexPartition ip;
  LogicalPartition lp;
  Domain launch_domain;
};

struct OscChainArgs {
  coord_t n_global;
};

inline state_type create_state(Runtime* runtime, Context ctx, std::size_t N, std::size_t G) {
  assert(runtime != nullptr);
  assert(N > 0);
  assert(G > 0);

  state_type st;
  st.runtime = runtime;
  st.ctx = ctx;
  st.N = N;
  st.G = G;
  st.M = (N + G - 1) / G;

  st.is = runtime->create_index_space(ctx, Rect<1>(0, static_cast<coord_t>(N - 1)));
  st.fs = runtime->create_field_space(ctx);

  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, st.fs);
    allocator.allocate_field(sizeof(double), FID_VALUE);
  }

  st.lr = runtime->create_logical_region(ctx, st.is, st.fs);
  st.launch_domain = Domain(Rect<1>(0, static_cast<coord_t>(st.M - 1)));

  DomainPointColoring coloring;
  for (std::size_t b = 0; b < st.M; ++b) {
    const coord_t lo = static_cast<coord_t>(b * G);
    const coord_t hi = static_cast<coord_t>(std::min<std::size_t>((b + 1) * G, N) - 1);
    coloring[DomainPoint::from_point<1>(Point<1>(static_cast<coord_t>(b)))] = Domain(Rect<1>(lo, hi));
  }

  st.ip = runtime->create_index_partition(ctx, st.is, st.launch_domain, coloring, true /*disjoint*/);
  st.lp = runtime->get_logical_partition(ctx, st.lr, st.ip);

  return st;
}

inline void destroy_state(state_type& st) {
  if (!st.runtime) return;
  st.runtime->destroy_logical_region(st.ctx, st.lr);
  st.runtime->destroy_field_space(st.ctx, st.fs);
  st.runtime->destroy_index_partition(st.ctx, st.ip);
  st.runtime->destroy_index_space(st.ctx, st.is);
  st = state_type{};
}

inline void write_state(state_type& st, const dvec& values) {
  assert(values.size() == st.N);

  RegionRequirement rr(st.lr, WRITE_DISCARD, EXCLUSIVE, st.lr);
  rr.add_field(FID_VALUE);

  InlineLauncher launcher(rr);
  PhysicalRegion pr = st.runtime->map_region(st.ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VALUE);
  Rect<1> rect = st.runtime->get_index_space_domain(st.ctx, st.is);

  std::size_t idx = 0;
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    acc[*pir] = values[idx++];
  }

  st.runtime->unmap_region(st.ctx, pr);
}

inline void read_state(const state_type& st, dvec& values) {
  values.resize(st.N);

  RegionRequirement rr(st.lr, READ_ONLY, EXCLUSIVE, st.lr);
  rr.add_field(FID_VALUE);

  InlineLauncher launcher(rr);
  PhysicalRegion pr = st.runtime->map_region(st.ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VALUE);
  Rect<1> rect = st.runtime->get_index_space_domain(st.ctx, st.is);

  std::size_t idx = 0;
  for (PointInRectIterator<1> pir(rect); pir(); pir++) {
    values[idx++] = acc[*pir];
  }

  st.runtime->unmap_region(st.ctx, pr);
}

inline void osc_chain_block_task(const Task* task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx,
                                 Runtime* runtime) {
  assert(task->arglen == sizeof(OscChainArgs));
  const auto* args = static_cast<const OscChainArgs*>(task->args);

  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VALUE);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VALUE);

  const Rect<1> my_rect =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

  const coord_t last = args->n_global - 1;

  for (PointInRectIterator<1> pir(my_rect); pir(); pir++) {
    const Point<1> p = *pir;
    const coord_t i = p[0];

    const double qi = q_acc[p];

    const double coupling_l =
        (i == 0) ? 0.5 * signed_pow(qi, LAMBDA - 1.0)
                 : signed_pow(qi - q_acc[Point<1>(i - 1)], LAMBDA - 1.0);

    const double coupling_r =
        (i == last) ? 0.5 * signed_pow(qi, LAMBDA - 1.0)
                    : signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[p] = -signed_pow(qi, KAPPA - 1.0) - coupling_l - coupling_r;
  }
}

inline void register_system_tasks() {
  static bool registered = false;
  if (registered) return;

  TaskVariantRegistrar registrar(OSC_CHAIN_BLOCK_TASK_ID, "osc_chain_block_task");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.set_leaf(true);
  Runtime::preregister_task_variant<osc_chain_block_task>(registrar, "osc_chain_block_task");

  registered = true;
}

inline FutureMap launch_osc_chain(state_type& q, state_type& dpdt) {
  assert(q.runtime == dpdt.runtime);
  assert(q.ctx == dpdt.ctx);
  assert(q.N == dpdt.N);
  assert(q.M == dpdt.M);

  OscChainArgs args{static_cast<coord_t>(q.N)};
  IndexLauncher launcher(OSC_CHAIN_BLOCK_TASK_ID, q.launch_domain,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(q.lr, READ_ONLY, EXCLUSIVE, q.lr));
  launcher.region_requirements[0].add_field(FID_VALUE);

  launcher.add_region_requirement(RegionRequirement(dpdt.lp, 0 /*identity projection*/,
                                                    WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
  launcher.region_requirements[1].add_field(FID_VALUE);

  return q.runtime->execute_index_space(q.ctx, launcher);
}

// Non-blocking variant (analogous to original osc_chain without global barrier)
inline void osc_chain(state_type& q, state_type& dpdt) {
  (void)launch_osc_chain(q, dpdt);
}

// Blocking variant (analogous to osc_chain_gb)
inline void osc_chain_gb(state_type& q, state_type& dpdt) {
  FutureMap fm = launch_osc_chain(q, dpdt);
  fm.wait_all_results();
}

}  // namespace legion_osc_chain

inline double energy(const dvec& q, const dvec& p) {
  using checked_math::pow;
  using std::abs;
  const std::size_t N = q.size();

  double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i]
       + pow(q[i], KAPPA) / KAPPA
       + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;

  return e;
}

inline double energy(const legion_osc_chain::state_type& q_st,
                     const legion_osc_chain::state_type& p_st) {
  dvec q, p;
  legion_osc_chain::read_state(q_st, q);
  legion_osc_chain::read_state(p_st, p);
  return energy(q, p);
}

#endif
