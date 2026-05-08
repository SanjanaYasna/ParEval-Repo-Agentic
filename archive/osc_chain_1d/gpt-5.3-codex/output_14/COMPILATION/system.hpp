// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion tasks/futures
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <stdexcept>
#include <cstdint>

#include <boost/math/special_functions/sign.hpp>
#include <legion.h>

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

namespace checked_math {
inline double pow(double x, double y)
{
    if (x == 0.0) return 0.0;
    using std::abs;
    using std::pow;
    return pow(abs(x), y);
}
}

inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// Legion version: each Future stores a uint64_t ID into a local block store.
typedef std::vector<Legion::Future> state_type;

namespace legion_osc_detail {

enum TaskIDs : Legion::TaskID {
    SYSTEM_BLOCK_TASK_ID = 10001
};

enum class BlockKind : int {
    FIRST  = 0,
    CENTER = 1,
    LAST   = 2,
    SINGLE = 3
};

struct SystemTaskArgs {
    BlockKind kind;
};

inline Legion::Runtime *g_runtime = nullptr;
inline Legion::Context  g_context = nullptr;

struct block_store {
    static uint64_t put(const dvec &v)
    {
        std::lock_guard<std::mutex> lock(mutex());
        const uint64_t id = next_id()++;
        data()[id] = std::make_shared<dvec>(v);
        return id;
    }

    static uint64_t put(shared_vec v)
    {
        std::lock_guard<std::mutex> lock(mutex());
        const uint64_t id = next_id()++;
        data()[id] = std::move(v);
        return id;
    }

    static shared_vec get(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(mutex());
        auto it = data().find(id);
        if (it == data().end()) {
            throw std::runtime_error("Invalid block_store id");
        }
        return it->second;
    }

private:
    static std::unordered_map<uint64_t, shared_vec> &data()
    {
        static std::unordered_map<uint64_t, shared_vec> s_data;
        return s_data;
    }

    static std::mutex &mutex()
    {
        static std::mutex s_mutex;
        return s_mutex;
    }

    static std::atomic<uint64_t> &next_id()
    {
        static std::atomic<uint64_t> s_next{1};
        return s_next;
    }
};

inline void compute_first(const dvec &q, double q_r, dvec &dpdt)
{
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_center(const dvec &q, double q_l, double q_r, dvec &dpdt)
{
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_last(const dvec &q, double q_l, dvec &dpdt)
{
    const size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1);
}

inline uint64_t system_block_task(const Legion::Task *task,
                                  const std::vector<Legion::PhysicalRegion> &,
                                  Legion::Context,
                                  Legion::Runtime *)
{
    const auto *args = static_cast<const SystemTaskArgs *>(task->args);
    if (args == nullptr) {
        throw std::runtime_error("system_block_task: missing task args");
    }

    // Future payloads are block IDs in block_store.
    // Layout by mode:
    // FIRST : q, q_right, dpdt
    // CENTER: q, q_left, q_right, dpdt
    // LAST  : q, q_left, dpdt
    // SINGLE: q, dpdt
    auto get_id = [&](size_t idx) -> uint64_t {
        return task->futures[idx].get_result<uint64_t>();
    };

    switch (args->kind) {
    case BlockKind::FIRST: {
        const shared_vec q      = block_store::get(get_id(0));
        const shared_vec q_right= block_store::get(get_id(1));
        const uint64_t dpdt_id  = get_id(2);
        shared_vec dpdt         = block_store::get(dpdt_id);

        const double q_r = q_right->empty() ? 0.0 : (*q_right)[0];
        compute_first(*q, q_r, *dpdt);
        return dpdt_id;
    }
    case BlockKind::CENTER: {
        const shared_vec q      = block_store::get(get_id(0));
        const shared_vec q_left = block_store::get(get_id(1));
        const shared_vec q_right= block_store::get(get_id(2));
        const uint64_t dpdt_id  = get_id(3);
        shared_vec dpdt         = block_store::get(dpdt_id);

        const double q_l = q_left->empty()  ? 0.0 : q_left->back();
        const double q_r = q_right->empty() ? 0.0 : (*q_right)[0];
        compute_center(*q, q_l, q_r, *dpdt);
        return dpdt_id;
    }
    case BlockKind::LAST: {
        const shared_vec q      = block_store::get(get_id(0));
        const shared_vec q_left = block_store::get(get_id(1));
        const uint64_t dpdt_id  = get_id(2);
        shared_vec dpdt         = block_store::get(dpdt_id);

        const double q_l = q_left->empty() ? 0.0 : q_left->back();
        compute_last(*q, q_l, *dpdt);
        return dpdt_id;
    }
    case BlockKind::SINGLE: {
        const shared_vec q      = block_store::get(get_id(0));
        const uint64_t dpdt_id  = get_id(1);
        shared_vec dpdt         = block_store::get(dpdt_id);

        compute_first(*q, 0.0, *dpdt); // both boundaries at 0 for one block case
        return dpdt_id;
    }
    }

    throw std::runtime_error("system_block_task: unknown block kind");
}

} // namespace legion_osc_detail

inline void set_system_legion_context(Legion::Runtime *runtime, Legion::Context context)
{
    legion_osc_detail::g_runtime = runtime;
    legion_osc_detail::g_context = context;
}

inline void register_system_tasks()
{
    static bool registered = false;
    if (registered) return;

    Legion::TaskVariantRegistrar registrar(legion_osc_detail::SYSTEM_BLOCK_TASK_ID,
                                           "system_block_task");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));

    Legion::Runtime::preregister_task_variant<uint64_t,
                                              legion_osc_detail::system_block_task>(
        registrar, "system_block_task");
    registered = true;
}

inline uint64_t store_state_block(const dvec &v)
{
    return legion_osc_detail::block_store::put(v);
}

inline uint64_t store_state_block(shared_vec v)
{
    return legion_osc_detail::block_store::put(std::move(v));
}

inline shared_vec load_state_block(uint64_t id)
{
    return legion_osc_detail::block_store::get(id);
}

inline void osc_chain(state_type &q, state_type &dpdt)
{
    using namespace Legion;
    if (legion_osc_detail::g_runtime == nullptr || legion_osc_detail::g_context == nullptr) {
        throw std::runtime_error("set_system_legion_context() must be called before osc_chain()");
    }

    const size_t N = q.size();
    if (N == 0) return;
    dpdt.resize(N);

    auto launch = [&](legion_osc_detail::BlockKind kind,
                      const std::vector<Future> &inputs) -> Future {
        legion_osc_detail::SystemTaskArgs args{kind};
        TaskLauncher launcher(legion_osc_detail::SYSTEM_BLOCK_TASK_ID,
                              TaskArgument(&args, sizeof(args)));
        for (const auto &f : inputs) launcher.add_future(f);
        return legion_osc_detail::g_runtime->execute_task(legion_osc_detail::g_context, launcher);
    };

    if (N == 1) {
        dpdt[0] = launch(legion_osc_detail::BlockKind::SINGLE, {q[0], dpdt[0]});
        return;
    }

    // first
    dpdt[0] = launch(legion_osc_detail::BlockKind::FIRST, {q[0], q[1], dpdt[0]});

    // middle
    for (size_t i = 1; i < N - 1; ++i) {
        dpdt[i] = launch(legion_osc_detail::BlockKind::CENTER,
                         {q[i], q[i - 1], q[i + 1], dpdt[i]});
    }

    // last
    dpdt[N - 1] = launch(legion_osc_detail::BlockKind::LAST,
                         {q[N - 1], q[N - 2], dpdt[N - 1]});
}

inline void osc_chain_gb(state_type &q, state_type &dpdt)
{
    osc_chain(q, dpdt);
    for (auto &f : dpdt) {
        (void)f.get_result<uint64_t>(); // global barrier
    }
}

inline double energy(const dvec &q, const dvec &p)
{
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
double energy(const S &q_fut, const S &p_fut)
{
    dvec q, p;
    q.reserve(q_fut.size());
    p.reserve(p_fut.size());

    for (size_t i = 0; i < q_fut.size(); ++i) {
        const uint64_t qid = q_fut[i].template get_result<uint64_t>();
        const uint64_t pid = p_fut[i].template get_result<uint64_t>();

        shared_vec q_blk = legion_osc_detail::block_store::get(qid);
        shared_vec p_blk = legion_osc_detail::block_store::get(pid);

        q.insert(q.end(), q_blk->begin(), q_blk->end());
        p.insert(p.end(), p_blk->begin(), p_blk->end());
    }
    return energy(q, p);
}

#endif // SYSTEM_HPP
