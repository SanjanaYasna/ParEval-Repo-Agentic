// Translated from HPX to Legion execution model
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <vector>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <algorithm>

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y) {
    if (x == 0.0) return 0.0;
    using std::abs;
    using std::pow;
    return pow(abs(x), y);
}
} // namespace checked_math

inline double signed_pow(double x, double k) {
    const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow(x, k) * s;
}

using dvec = std::vector<double>;
using state_type = std::vector<Legion::Future>;

namespace legion_osc_chain {

inline Legion::Runtime *g_runtime = nullptr;
inline Legion::Context g_context = nullptr;

inline void set_runtime_context(Legion::Runtime *rt, Legion::Context ctx) {
    g_runtime = rt;
    g_context = ctx;
}

inline void serialize_vec(Legion::Serializer &ser, const dvec &v) {
    const size_t n = v.size();
    ser.serialize(n);
    if (n) ser.serialize(v.data(), n * sizeof(double));
}

inline dvec deserialize_vec(Legion::Deserializer &dez) {
    size_t n = 0;
    dez.deserialize(n);
    dvec v(n);
    if (n) dez.deserialize(v.data(), n * sizeof(double));
    return v;
}

inline dvec future_to_vec(const Legion::Future &f) {
    const void *ptr = f.get_untyped_pointer();
    const size_t bytes = f.get_untyped_size();
    Legion::Deserializer dez(ptr, bytes);
    return deserialize_vec(dez);
}

inline void set_return_vec(Legion::Runtime *runtime, Legion::Context ctx, const dvec &v) {
    Legion::Serializer ser;
    serialize_vec(ser, v);
    runtime->set_return_value(ctx, ser.get_buffer(), ser.get_used_bytes());
}

inline dvec compute_center_block(const dvec &q, double q_l, double q_r) {
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
    return dpdt;
}

inline dvec compute_first_block(const dvec &q, double q_r) {
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
    return dpdt;
}

inline dvec compute_last_block(const dvec &q, double q_l) {
    const size_t N = q.size();
    dvec dpdt(N, 0.0);
    if (N == 0) return dpdt;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1);
    return dpdt;
}

enum : Legion::TaskID {
    SYSTEM_FIRST_BLOCK_TASK_ID  = 10001,
    SYSTEM_CENTER_BLOCK_TASK_ID = 10002,
    SYSTEM_LAST_BLOCK_TASK_ID   = 10003,
    SYSTEM_SINGLE_BLOCK_TASK_ID = 10004
};

inline void system_first_block_task(const Legion::Task *task,
                                    const std::vector<Legion::PhysicalRegion> &,
                                    Legion::Context ctx, Legion::Runtime *runtime) {
    assert(task->futures.size() == 2);
    const dvec q = future_to_vec(task->futures[0]);
    const dvec q_right = future_to_vec(task->futures[1]);
    const double q_r = q_right.empty() ? 0.0 : q_right.front();
    set_return_vec(runtime, ctx, compute_first_block(q, q_r));
}

inline void system_center_block_task(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &,
                                     Legion::Context ctx, Legion::Runtime *runtime) {
    assert(task->futures.size() == 3);
    const dvec q = future_to_vec(task->futures[0]);
    const dvec q_left = future_to_vec(task->futures[1]);
    const dvec q_right = future_to_vec(task->futures[2]);
    const double q_l = q_left.empty() ? 0.0 : q_left.back();
    const double q_r = q_right.empty() ? 0.0 : q_right.front();
    set_return_vec(runtime, ctx, compute_center_block(q, q_l, q_r));
}

inline void system_last_block_task(const Legion::Task *task,
                                   const std::vector<Legion::PhysicalRegion> &,
                                   Legion::Context ctx, Legion::Runtime *runtime) {
    assert(task->futures.size() == 2);
    const dvec q = future_to_vec(task->futures[0]);
    const dvec q_left = future_to_vec(task->futures[1]);
    const double q_l = q_left.empty() ? 0.0 : q_left.back();
    set_return_vec(runtime, ctx, compute_last_block(q, q_l));
}

inline void system_single_block_task(const Legion::Task *task,
                                     const std::vector<Legion::PhysicalRegion> &,
                                     Legion::Context ctx, Legion::Runtime *runtime) {
    assert(task->futures.size() == 1);
    const dvec q = future_to_vec(task->futures[0]);
    set_return_vec(runtime, ctx, compute_center_block(q, 0.0, 0.0));
}

inline bool g_registered = false;

inline void register_system_tasks() {
    if (g_registered) return;

    {
        Legion::TaskVariantRegistrar r(SYSTEM_FIRST_BLOCK_TASK_ID, "system_first_block_task");
        r.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<system_first_block_task>(r, "system_first_block_task");
    }
    {
        Legion::TaskVariantRegistrar r(SYSTEM_CENTER_BLOCK_TASK_ID, "system_center_block_task");
        r.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<system_center_block_task>(r, "system_center_block_task");
    }
    {
        Legion::TaskVariantRegistrar r(SYSTEM_LAST_BLOCK_TASK_ID, "system_last_block_task");
        r.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<system_last_block_task>(r, "system_last_block_task");
    }
    {
        Legion::TaskVariantRegistrar r(SYSTEM_SINGLE_BLOCK_TASK_ID, "system_single_block_task");
        r.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<system_single_block_task>(r, "system_single_block_task");
    }

    g_registered = true;
}

inline void wait_all(const state_type &futs) {
    for (const auto &f : futs) {
        (void)f.get_untyped_pointer();
    }
}

} // namespace legion_osc_chain

inline void osc_chain(state_type &q, state_type &dpdt) {
    using namespace legion_osc_chain;
    assert(g_runtime != nullptr && g_context != nullptr);

    const size_t N = q.size();
    if (N == 0) return;
    dpdt.resize(N);

    if (N == 1) {
        Legion::TaskLauncher single(SYSTEM_SINGLE_BLOCK_TASK_ID, Legion::TaskArgument(nullptr, 0));
        single.add_future(q[0]);
        dpdt[0] = g_runtime->execute_task(g_context, single);
        return;
    }

    {
        Legion::TaskLauncher first(SYSTEM_FIRST_BLOCK_TASK_ID, Legion::TaskArgument(nullptr, 0));
        first.add_future(q[0]);
        first.add_future(q[1]);
        dpdt[0] = g_runtime->execute_task(g_context, first);
    }

    for (size_t i = 1; i < N - 1; ++i) {
        Legion::TaskLauncher center(SYSTEM_CENTER_BLOCK_TASK_ID, Legion::TaskArgument(nullptr, 0));
        center.add_future(q[i]);
        center.add_future(q[i - 1]);
        center.add_future(q[i + 1]);
        dpdt[i] = g_runtime->execute_task(g_context, center);
    }

    {
        Legion::TaskLauncher last(SYSTEM_LAST_BLOCK_TASK_ID, Legion::TaskArgument(nullptr, 0));
        last.add_future(q[N - 1]);
        last.add_future(q[N - 2]);
        dpdt[N - 1] = g_runtime->execute_task(g_context, last);
    }
}

inline void osc_chain_gb(state_type &q, state_type &dpdt) {
    osc_chain(q, dpdt);
    legion_osc_chain::wait_all(dpdt);
}

inline double energy(const dvec &q, const dvec &p) {
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i]
          +  pow(q[i], KAPPA) / KAPPA
          +  pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
      +  pow(q[N - 1], KAPPA) / KAPPA
      + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

template <typename S>
double energy(const S &q_fut, const S &p_fut) {
    dvec q, p;
    for (size_t i = 0; i < q_fut.size(); ++i) {
        dvec qb = legion_osc_chain::future_to_vec(q_fut[i]);
        dvec pb = legion_osc_chain::future_to_vec(p_fut[i]);
        q.insert(q.end(), qb.begin(), qb.end());
        p.insert(p.end(), pb.begin(), pb.end());
    }
    return energy(q, p);
}

#endif
