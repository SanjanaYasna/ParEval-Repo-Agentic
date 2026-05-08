// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion task/future model.

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>
#include <legion_stl.h>

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}
} // namespace checked_math

inline double signed_pow(double x, double k) {
  using std::copysign;
  return copysign(checked_math::pow(x, k), x);
}

typedef std::vector<double> dvec;
typedef std::vector<Legion::Future> state_type;

namespace legion_system_detail {

enum TaskIDs : Legion::TaskID {
  TASK_GET_FIRST_VALUE = 50001,
  TASK_GET_LAST_VALUE  = 50002,
  TASK_SYSTEM_FIRST    = 50003,
  TASK_SYSTEM_CENTER   = 50004,
  TASK_SYSTEM_LAST     = 50005
};

inline Legion::Runtime *g_runtime = nullptr;
inline Legion::Context g_context = nullptr;

// --- Boundary extraction tasks ---

inline double get_first_value_task(const Legion::Task *task,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context,
                                   Legion::Runtime *) {
  const dvec q = task->futures[0].get_result<dvec>();
  return q.empty() ? 0.0 : q.front();
}

inline double get_last_value_task(const Legion::Task *task,
                                  const std::vector<Legion::PhysicalRegion> &,
                                  Legion::Context,
                                  Legion::Runtime *) {
  const dvec q = task->futures[0].get_result<dvec>();
  return q.empty() ? 0.0 : q.back();
}

// --- System block tasks ---

inline dvec system_first_block_task(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &,
                                    Legion::Context,
                                    Legion::Runtime *) {
  const dvec q = task->futures[0].get_result<dvec>();
  const double q_r = task->futures[1].get_result<double>();

  const size_t N = q.size();
  dvec dpdt(N, 0.0);
  if (N == 0) return dpdt;

  double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
  for (size_t i = 0; i < N - 1; ++i) {
    dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
    coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
    dpdt[i] -= coupling_lr;
  }

  dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
              + coupling_lr
              - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
  return dpdt;
}

inline dvec system_center_block_task(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &,
                                     Legion::Context,
                                     Legion::Runtime *) {
  const dvec q = task->futures[0].get_result<dvec>();
  const double q_l = task->futures[1].get_result<double>();
  const double q_r = task->futures[2].get_result<double>();

  const size_t N = q.size();
  dvec dpdt(N, 0.0);
  if (N == 0) return dpdt;

  double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
  for (size_t i = 0; i < N - 1; ++i) {
    dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
    coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
    dpdt[i] -= coupling_lr;
  }

  dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
              + coupling_lr
              - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
  return dpdt;
}

inline dvec system_last_block_task(const Legion::Task *task,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context,
                                   Legion::Runtime *) {
  const dvec q = task->futures[0].get_result<dvec>();
  const double q_l = task->futures[1].get_result<double>();

  const size_t N = q.size();
  dvec dpdt(N, 0.0);
  if (N == 0) return dpdt;

  double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
  for (size_t i = 0; i < N - 1; ++i) {
    dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
    coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
    dpdt[i] -= coupling_lr;
  }

  dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
              + coupling_lr
              - signed_pow(q[N - 1], LAMBDA - 1.0);
  return dpdt;
}

inline Legion::Future launch_get_first(const Legion::Future &q_block) {
  Legion::TaskLauncher launcher(TASK_GET_FIRST_VALUE, Legion::TaskArgument(nullptr, 0));
  launcher.add_future(q_block);
  return g_runtime->execute_task(g_context, launcher);
}

inline Legion::Future launch_get_last(const Legion::Future &q_block) {
  Legion::TaskLauncher launcher(TASK_GET_LAST_VALUE, Legion::TaskArgument(nullptr, 0));
  launcher.add_future(q_block);
  return g_runtime->execute_task(g_context, launcher);
}

} // namespace legion_system_detail

inline void set_system_legion_context(Legion::Runtime *runtime, Legion::Context context) {
  legion_system_detail::g_runtime = runtime;
  legion_system_detail::g_context = context;
}

inline void preregister_system_tasks() {
  using namespace Legion;
  using namespace legion_system_detail;

  static bool registered = false;
  if (registered) return;
  registered = true;

  {
    TaskVariantRegistrar r(TASK_GET_FIRST_VALUE, "get_first_value_task");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    r.set_leaf();
    Runtime::preregister_task_variant<double, get_first_value_task>(r, "get_first_value_task");
  }
  {
    TaskVariantRegistrar r(TASK_GET_LAST_VALUE, "get_last_value_task");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    r.set_leaf();
    Runtime::preregister_task_variant<double, get_last_value_task>(r, "get_last_value_task");
  }
  {
    TaskVariantRegistrar r(TASK_SYSTEM_FIRST, "system_first_block_task");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<dvec, system_first_block_task>(r, "system_first_block_task");
  }
  {
    TaskVariantRegistrar r(TASK_SYSTEM_CENTER, "system_center_block_task");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<dvec, system_center_block_task>(r, "system_center_block_task");
  }
  {
    TaskVariantRegistrar r(TASK_SYSTEM_LAST, "system_last_block_task");
    r.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<dvec, system_last_block_task>(r, "system_last_block_task");
  }
}

inline void osc_chain(state_type &q, state_type &dpdt) {
  using namespace Legion;
  using namespace legion_system_detail;

  if (g_runtime == nullptr || g_context == nullptr) {
    throw std::runtime_error("set_system_legion_context must be called before osc_chain.");
  }

  const size_t N = q.size();
  if (N == 0) return;

  if (N == 1) {
    // Single block: both external boundaries are zero (equivalent to center with q_l=q_r=0).
    Future q_l = Future::from_value<double>(g_runtime, 0.0);
    Future q_r = Future::from_value<double>(g_runtime, 0.0);
    TaskLauncher center(TASK_SYSTEM_CENTER, TaskArgument(nullptr, 0));
    center.add_future(q[0]);
    center.add_future(q_l);
    center.add_future(q_r);
    dpdt[0] = g_runtime->execute_task(g_context, center);
    return;
  }

  // First block
  {
    Future q_r = launch_get_first(q[1]);
    TaskLauncher first(TASK_SYSTEM_FIRST, TaskArgument(nullptr, 0));
    first.add_future(q[0]);
    first.add_future(q_r);
    dpdt[0] = g_runtime->execute_task(g_context, first);
  }

  // Middle blocks
  for (size_t i = 1; i < N - 1; ++i) {
    Future q_l = launch_get_last(q[i - 1]);
    Future q_r = launch_get_first(q[i + 1]);

    TaskLauncher center(TASK_SYSTEM_CENTER, TaskArgument(nullptr, 0));
    center.add_future(q[i]);
    center.add_future(q_l);
    center.add_future(q_r);
    dpdt[i] = g_runtime->execute_task(g_context, center);
  }

  // Last block
  {
    Future q_l = launch_get_last(q[N - 2]);
    TaskLauncher last(TASK_SYSTEM_LAST, TaskArgument(nullptr, 0));
    last.add_future(q[N - 1]);
    last.add_future(q_l);
    dpdt[N - 1] = g_runtime->execute_task(g_context, last);
  }
}

inline void osc_chain_gb(state_type &q, state_type &dpdt) {
  osc_chain(q, dpdt);
  for (auto &f : dpdt) {
    (void)f.get_result<dvec>(); // global barrier
  }
}

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

template <typename S>
double energy(const S &q_fut, const S &p_fut) {
  dvec q, p;
  for (size_t i = 0; i < q_fut.size(); ++i) {
    dvec qb = q_fut[i].template get_result<dvec>();
    dvec pb = p_fut[i].template get_result<dvec>();
    q.insert(q.end(), qb.begin(), qb.end());
    p.insert(p.end(), pb.begin(), pb.end());
  }
  return energy(q, p);
}

#endif
