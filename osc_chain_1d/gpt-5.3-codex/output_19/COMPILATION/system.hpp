// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion tasks/futures.
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <boost/math/special_functions/sign.hpp>

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
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
typedef std::shared_ptr<dvec> shared_vec;

// Legion state: each block is a Future containing serialized data:
// [size_t n][double data0...data(n-1)]
typedef std::vector<Legion::Future> state_type;

namespace legion_system_detail {

enum TaskIDs : Legion::TaskID {
    EXTRACT_FIRST_VALUE_TASK_ID = 20001,
    EXTRACT_LAST_VALUE_TASK_ID  = 20002,
    SYSTEM_FIRST_BLOCK_TASK_ID  = 20003,
    SYSTEM_CENTER_BLOCK_TASK_ID = 20004,
    SYSTEM_LAST_BLOCK_TASK_ID   = 20005
};

struct RuntimeContext {
    Legion::Runtime* runtime = nullptr;
    Legion::Context context{};
    bool initialized = false;
};

inline RuntimeContext& runtime_context() {
    static RuntimeContext rc;
    return rc;
}

inline dvec unpack_vector_buffer(const void* ptr, size_t bytes) {
    if (ptr == nullptr || bytes < sizeof(size_t)) {
        throw std::runtime_error("Invalid serialized vector buffer.");
    }

    size_t n = 0;
    std::memcpy(&n, ptr, sizeof(size_t));

    const size_t expected = sizeof(size_t) + n * sizeof(double);
    if (bytes != expected) {
        throw std::runtime_error("Serialized vector size mismatch.");
    }

    dvec v(n);
    if (n > 0) {
        const char* base = static_cast<const char*>(ptr);
        std::memcpy(v.data(), base + sizeof(size_t), n * sizeof(double));
    }
    return v;
}

inline dvec unpack_vector_future(const Legion::Future& f) {
    size_t bytes = 0;
    const void* ptr = f.get_untyped_pointer(true /*silence warnings*/, &bytes);
    return unpack_vector_buffer(ptr, bytes);
}

inline Legion::TaskResult pack_vector_result(const dvec& v) {
    const size_t n = v.size();
    const size_t bytes = sizeof(size_t) + n * sizeof(double);
    char* buffer = static_cast<char*>(std::malloc(bytes));
    if (buffer == nullptr) {
        throw std::bad_alloc();
    }

    std::memcpy(buffer, &n, sizeof(size_t));
    if (n > 0) {
        std::memcpy(buffer + sizeof(size_t), v.data(), n * sizeof(double));
    }

    // Legion takes ownership when own=true.
    return Legion::TaskResult(buffer, bytes, true /*own*/);
}

inline void compute_first_block(const dvec& q, double q_r, dvec& dpdt) {
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] =
        -signed_pow(q[N - 1], KAPPA - 1) + coupling_lr -
        signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_center_block(const dvec& q, double q_l, double q_r, dvec& dpdt) {
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] =
        -signed_pow(q[N - 1], KAPPA - 1) + coupling_lr -
        signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_last_block(const dvec& q, double q_l, dvec& dpdt) {
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] =
        -signed_pow(q[N - 1], KAPPA - 1) + coupling_lr -
        signed_pow(q[N - 1], LAMBDA - 1);
}

inline double extract_first_value_task(const Legion::Task* task,
                                       const std::vector<Legion::PhysicalRegion>&,
                                       Legion::Context,
                                       Legion::Runtime*) {
    assert(task->futures.size() == 1);
    const dvec q = unpack_vector_future(task->futures[0]);
    return q.empty() ? 0.0 : q.front();
}

inline double extract_last_value_task(const Legion::Task* task,
                                      const std::vector<Legion::PhysicalRegion>&,
                                      Legion::Context,
                                      Legion::Runtime*) {
    assert(task->futures.size() == 1);
    const dvec q = unpack_vector_future(task->futures[0]);
    return q.empty() ? 0.0 : q.back();
}

inline Legion::TaskResult system_first_block_task(const Legion::Task* task,
                                                  const std::vector<Legion::PhysicalRegion>&,
                                                  Legion::Context,
                                                  Legion::Runtime*) {
    assert(task->futures.size() == 2);
    const dvec q = unpack_vector_future(task->futures[0]);
    const double q_r = task->futures[1].get_result<double>();

    dvec out;
    compute_first_block(q, q_r, out);
    return pack_vector_result(out);
}

inline Legion::TaskResult system_center_block_task(const Legion::Task* task,
                                                   const std::vector<Legion::PhysicalRegion>&,
                                                   Legion::Context,
                                                   Legion::Runtime*) {
    assert(task->futures.size() == 3);
    const dvec q = unpack_vector_future(task->futures[0]);
    const double q_l = task->futures[1].get_result<double>();
    const double q_r = task->futures[2].get_result<double>();

    dvec out;
    compute_center_block(q, q_l, q_r, out);
    return pack_vector_result(out);
}

inline Legion::TaskResult system_last_block_task(const Legion::Task* task,
                                                 const std::vector<Legion::PhysicalRegion>&,
                                                 Legion::Context,
                                                 Legion::Runtime*) {
    assert(task->futures.size() == 2);
    const dvec q = unpack_vector_future(task->futures[0]);
    const double q_l = task->futures[1].get_result<double>();

    dvec out;
    compute_last_block(q, q_l, out);
    return pack_vector_result(out);
}

}  // namespace legion_system_detail

inline void set_legion_runtime_context(Legion::Context ctx, Legion::Runtime* runtime) {
    auto& rc = legion_system_detail::runtime_context();
    rc.context = ctx;
    rc.runtime = runtime;
    rc.initialized = true;
}

inline void preregister_system_tasks() {
    using namespace Legion;
    using namespace legion_system_detail;

    static bool registered = false;
    if (registered) return;
    registered = true;

    {
        TaskVariantRegistrar registrar(EXTRACT_FIRST_VALUE_TASK_ID, "extract_first_value_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<extract_first_value_task>(registrar, "extract_first_value_task");
    }
    {
        TaskVariantRegistrar registrar(EXTRACT_LAST_VALUE_TASK_ID, "extract_last_value_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<extract_last_value_task>(registrar, "extract_last_value_task");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_FIRST_BLOCK_TASK_ID, "system_first_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_first_block_task>(registrar, "system_first_block_task");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_CENTER_BLOCK_TASK_ID, "system_center_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_center_block_task>(registrar, "system_center_block_task");
    }
    {
        TaskVariantRegistrar registrar(SYSTEM_LAST_BLOCK_TASK_ID, "system_last_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<system_last_block_task>(registrar, "system_last_block_task");
    }
}

inline void wait_all(const state_type& futures) {
    for (const auto& f : futures) f.wait();
}

inline void osc_chain(state_type& q, state_type& dpdt) {
    using namespace Legion;
    using namespace legion_system_detail;

    auto& rc = runtime_context();
    if (!rc.initialized || rc.runtime == nullptr) {
        throw std::runtime_error(
            "Legion runtime/context not set. Call set_legion_runtime_context(ctx, runtime) first.");
    }

    Runtime* runtime = rc.runtime;
    Context ctx = rc.context;

    const size_t N = q.size();
    if (dpdt.size() != N) dpdt.resize(N);
    if (N == 0) return;

    if (N == 1) {
        TaskLauncher first(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        first.add_future(q[0]);
        first.add_future(Future::from_value(runtime, 0.0));
        dpdt[0] = runtime->execute_task(ctx, first);
        return;
    }

    // First block
    TaskLauncher q_r_first_launch(EXTRACT_FIRST_VALUE_TASK_ID, TaskArgument(nullptr, 0));
    q_r_first_launch.add_future(q[1]);
    Future q_r_first = runtime->execute_task(ctx, q_r_first_launch);

    TaskLauncher first(SYSTEM_FIRST_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
    first.add_future(q[0]);
    first.add_future(q_r_first);
    dpdt[0] = runtime->execute_task(ctx, first);

    // Center blocks
    for (size_t i = 1; i + 1 < N; ++i) {
        TaskLauncher q_l_launch(EXTRACT_LAST_VALUE_TASK_ID, TaskArgument(nullptr, 0));
        q_l_launch.add_future(q[i - 1]);
        Future q_l = runtime->execute_task(ctx, q_l_launch);

        TaskLauncher q_r_launch(EXTRACT_FIRST_VALUE_TASK_ID, TaskArgument(nullptr, 0));
        q_r_launch.add_future(q[i + 1]);
        Future q_r = runtime->execute_task(ctx, q_r_launch);

        TaskLauncher center(SYSTEM_CENTER_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
        center.add_future(q[i]);
        center.add_future(q_l);
        center.add_future(q_r);
        dpdt[i] = runtime->execute_task(ctx, center);
    }

    // Last block
    TaskLauncher q_l_last_launch(EXTRACT_LAST_VALUE_TASK_ID, TaskArgument(nullptr, 0));
    q_l_last_launch.add_future(q[N - 2]);
    Future q_l_last = runtime->execute_task(ctx, q_l_last_launch);

    TaskLauncher last(SYSTEM_LAST_BLOCK_TASK_ID, TaskArgument(nullptr, 0));
    last.add_future(q[N - 1]);
    last.add_future(q_l_last);
    dpdt[N - 1] = runtime->execute_task(ctx, last);
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
    osc_chain(q, dpdt);
    wait_all(dpdt);
}

inline double energy(const dvec& q, const dvec& p) {
    using checked_math::pow;
    using std::abs;

    const size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA +
             pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA +
         0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

template <typename S>
double energy(const S& q_fut, const S& p_fut) {
    dvec q, p;
    for (size_t i = 0; i < q_fut.size(); ++i) {
        dvec q_block = legion_system_detail::unpack_vector_future(q_fut[i]);
        dvec p_block = legion_system_detail::unpack_vector_future(p_fut[i]);
        q.insert(q.end(), q_block.begin(), q_block.end());
        p.insert(p.end(), p_block.begin(), p_block.end());
    }
    return energy(q, p);
}

#endif
