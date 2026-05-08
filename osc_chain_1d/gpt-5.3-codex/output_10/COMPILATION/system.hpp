// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion task/future model.

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>

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
typedef std::shared_ptr<dvec> shared_vec;

// In Legion translation, each block is represented as a Future containing a serialized dvec.
typedef std::vector<Legion::Future> state_type;

enum : Legion::TaskID {
  SYSTEM_BLOCK_TASK_ID = 1001
};

enum BlockKind : int {
  BLOCK_FIRST  = 0,
  BLOCK_CENTER = 1,
  BLOCK_LAST   = 2,
  BLOCK_SINGLE = 3
};

struct BlockTaskArgs {
  int kind;
};

inline dvec deserialize_dvec(const void* buffer, size_t bytes) {
  Legion::Deserializer dez(buffer, bytes);
  size_t n = 0;
  dez.deserialize(n);
  dvec out(n);
  for (size_t i = 0; i < n; ++i) dez.deserialize(out[i]);
  return out;
}

inline dvec future_to_dvec(const Legion::Future& f) {
  const void* buffer = f.get_untyped_pointer();
  const size_t bytes = f.get_untyped_size();
  return deserialize_dvec(buffer, bytes);
}

inline Legion::TaskResult dvec_to_task_result(const dvec& v) {
  Legion::Serializer rez;
  const size_t n = v.size();
  rez.serialize(n);
  for (size_t i = 0; i < n; ++i) rez.serialize(v[i]);

  void* out = std::malloc(rez.get_used_bytes());
  std::memcpy(out, rez.get_buffer(), rez.get_used_bytes());
  return Legion::TaskResult(out, rez.get_used_bytes(), true /*owned*/);
}

inline dvec compute_block_dpdt(const dvec& q, double q_l, double q_r) {
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

inline Legion::TaskResult system_block_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>&,
    Legion::Context,
    Legion::Runtime*) {
  const auto* args = static_cast<const BlockTaskArgs*>(task->args);
  const int kind = args->kind;

  // futures layout:
  // first  : [q_i, q_{i+1}]
  // center : [q_i, q_{i-1}, q_{i+1}]
  // last   : [q_i, q_{i-1}]
  // single : [q_i]
  dvec q = future_to_dvec(task->futures[0]);

  double q_l = 0.0, q_r = 0.0;
  if (kind == BLOCK_FIRST) {
    dvec qr = future_to_dvec(task->futures[1]);
    q_r = qr.front();
  } else if (kind == BLOCK_CENTER) {
    dvec ql = future_to_dvec(task->futures[1]);
    dvec qr = future_to_dvec(task->futures[2]);
    q_l = ql.back();
    q_r = qr.front();
  } else if (kind == BLOCK_LAST) {
    dvec ql = future_to_dvec(task->futures[1]);
    q_l = ql.back();
  }  // BLOCK_SINGLE keeps q_l=q_r=0

  dvec dpdt = compute_block_dpdt(q, q_l, q_r);
  return dvec_to_task_result(dpdt);
}

// Must be called before Runtime::start(...)
inline void preregister_system_tasks() {
  static bool done = false;
  if (done) return;
  done = true;

  Legion::TaskVariantRegistrar registrar(SYSTEM_BLOCK_TASK_ID, "system_block_task");
  registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
  registrar.set_leaf();
  Legion::Runtime::preregister_task_variant<system_block_task>(registrar, "system_block_task");
}

inline Legion::Context& system_ctx_ref() {
  static Legion::Context ctx = nullptr;
  return ctx;
}

inline Legion::Runtime*& system_rt_ref() {
  static Legion::Runtime* rt = nullptr;
  return rt;
}

// Must be called from a Legion task context (e.g., top-level task) before osc_chain usage.
inline void bind_system_runtime(Legion::Context ctx, Legion::Runtime* runtime) {
  system_ctx_ref() = ctx;
  system_rt_ref() = runtime;
}

inline void osc_chain(state_type& q, state_type& dpdt) {
  Legion::Runtime* runtime = system_rt_ref();
  Legion::Context ctx = system_ctx_ref();
  assert(runtime != nullptr && "bind_system_runtime must be called before osc_chain");
  assert(ctx != nullptr && "bind_system_runtime must be called before osc_chain");

  const size_t N = q.size();
  dpdt.resize(N);
  if (N == 0) return;

  for (size_t i = 0; i < N; ++i) {
    BlockTaskArgs args{};
    Legion::TaskLauncher launcher(
        SYSTEM_BLOCK_TASK_ID,
        Legion::TaskArgument(&args, sizeof(args)));

    if (N == 1) {
      args.kind = BLOCK_SINGLE;
      launcher = Legion::TaskLauncher(
          SYSTEM_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
      launcher.add_future(q[i]);
    } else if (i == 0) {
      args.kind = BLOCK_FIRST;
      launcher = Legion::TaskLauncher(
          SYSTEM_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
      launcher.add_future(q[i]);
      launcher.add_future(q[i + 1]);
    } else if (i + 1 == N) {
      args.kind = BLOCK_LAST;
      launcher = Legion::TaskLauncher(
          SYSTEM_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
      launcher.add_future(q[i]);
      launcher.add_future(q[i - 1]);
    } else {
      args.kind = BLOCK_CENTER;
      launcher = Legion::TaskLauncher(
          SYSTEM_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
      launcher.add_future(q[i]);
      launcher.add_future(q[i - 1]);
      launcher.add_future(q[i + 1]);
    }

    dpdt[i] = runtime->execute_task(ctx, launcher);
  }
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
  osc_chain(q, dpdt);
  for (auto& f : dpdt) {
    (void)f.get_untyped_size(); // wait
  }
}

inline double energy(const dvec& q, const dvec& p) {
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
double energy(const S& q_fut, const S& p_fut) {
  dvec q, p;
  for (size_t i = 0; i < q_fut.size(); ++i) {
    dvec q_blk = future_to_dvec(q_fut[i]);
    dvec p_blk = future_to_dvec(p_fut[i]);
    q.insert(q.end(), q_blk.begin(), q_blk.end());
    p.insert(p.end(), p_blk.begin(), p_blk.end());
  }
  return energy(q, p);
}

#endif
