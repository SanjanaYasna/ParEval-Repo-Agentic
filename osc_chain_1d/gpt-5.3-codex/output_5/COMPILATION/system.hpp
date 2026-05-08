// Copyright 2013 Mario Mulansky
// Legion translation of system.hpp

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <vector>
#include <memory>
#include <cmath>
#include <cstddef>
#include <cassert>

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
} // namespace checked_math

inline double signed_pow(double x, double k)
{
    const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_math::pow(x, k) * s;
}

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;
typedef std::vector<shared_vec> state_type;

namespace legion_system_detail {

enum : Legion::TaskID {
    OSC_CHAIN_BLOCK_TASK_ID = 10001
};

enum BlockKind : int {
    BLOCK_FIRST  = 0,
    BLOCK_CENTER = 1,
    BLOCK_LAST   = 2,
    BLOCK_SINGLE = 3
};

struct OscBlockTaskArgs {
    const double* q;
    double* dpdt;
    std::size_t n;
    double q_l;
    double q_r;
    int kind;
};

inline void compute_block_rhs(const OscBlockTaskArgs& a)
{
    if (a.n == 0 || a.q == nullptr || a.dpdt == nullptr) return;

    const double* q = a.q;
    double* dpdt = a.dpdt;
    const std::size_t N = a.n;

    double coupling_lr = 0.0;

    switch (a.kind) {
        case BLOCK_FIRST:
            coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
            break;
        case BLOCK_CENTER:
        case BLOCK_LAST:
        case BLOCK_SINGLE:
            coupling_lr = -signed_pow(q[0] - a.q_l, LAMBDA - 1.0);
            break;
        default:
            coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
            break;
    }

    for (std::size_t i = 0; i + 1 < N; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
        dpdt[i] -= coupling_lr;
    }

    if (a.kind == BLOCK_LAST) {
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                    + coupling_lr
                    - signed_pow(q[N - 1], LAMBDA - 1.0);
    } else {
        // FIRST / CENTER / SINGLE
        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                    + coupling_lr
                    - signed_pow(q[N - 1] - a.q_r, LAMBDA - 1.0);
    }
}

inline void osc_chain_block_task(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>&,
                                 Legion::Context,
                                 Legion::Runtime*)
{
    const auto* args = static_cast<const OscBlockTaskArgs*>(task->args);
    assert(args != nullptr);
    compute_block_rhs(*args);
}

inline Legion::Runtime*& runtime_ref()
{
    static thread_local Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& context_ref()
{
    static thread_local Legion::Context ctx = nullptr;
    return ctx;
}

inline OscBlockTaskArgs make_args(state_type& q, state_type& dpdt, std::size_t i)
{
    const std::size_t B = q.size();
    assert(i < B);
    assert(q[i] != nullptr);

    if (!dpdt[i]) dpdt[i] = std::make_shared<dvec>();
    dpdt[i]->resize(q[i]->size());

    OscBlockTaskArgs a{};
    a.q = q[i]->data();
    a.dpdt = dpdt[i]->data();
    a.n = q[i]->size();
    a.q_l = 0.0;
    a.q_r = 0.0;

    if (B == 1) {
        a.kind = BLOCK_SINGLE;
        return a;
    }

    if (i == 0) {
        a.kind = BLOCK_FIRST;
        a.q_r = (q[i + 1] && !q[i + 1]->empty()) ? (*(q[i + 1]))[0] : 0.0;
    } else if (i + 1 == B) {
        a.kind = BLOCK_LAST;
        a.q_l = (q[i - 1] && !q[i - 1]->empty()) ? (*(q[i - 1]))[q[i - 1]->size() - 1] : 0.0;
    } else {
        a.kind = BLOCK_CENTER;
        a.q_l = (q[i - 1] && !q[i - 1]->empty()) ? (*(q[i - 1]))[q[i - 1]->size() - 1] : 0.0;
        a.q_r = (q[i + 1] && !q[i + 1]->empty()) ? (*(q[i + 1]))[0] : 0.0;
    }

    return a;
}

} // namespace legion_system_detail

// Call once before Runtime::start(...)
inline void preregister_system_tasks()
{
    static bool registered = false;
    if (registered) return;

    Legion::TaskVariantRegistrar registrar(
        legion_system_detail::OSC_CHAIN_BLOCK_TASK_ID,
        "osc_chain_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();

    Legion::Runtime::preregister_task_variant<legion_system_detail::osc_chain_block_task>(
        registrar, "osc_chain_block_task");

    registered = true;
}

// Call from top-level Legion task before using osc_chain(...)
inline void set_system_legion_context(Legion::Runtime* runtime, Legion::Context ctx)
{
    legion_system_detail::runtime_ref() = runtime;
    legion_system_detail::context_ref() = ctx;
}

inline void osc_chain(state_type& q, state_type& dpdt)
{
    using namespace legion_system_detail;

    const std::size_t B = q.size();
    dpdt.resize(B);

    Legion::Runtime* runtime = runtime_ref();
    Legion::Context ctx = context_ref();

    // Fallback: serial execution if context not set.
    if (runtime == nullptr || ctx == nullptr) {
        for (std::size_t i = 0; i < B; ++i) {
            OscBlockTaskArgs args = make_args(q, dpdt, i);
            compute_block_rhs(args);
        }
        return;
    }

    std::vector<Legion::Future> done;
    done.reserve(B);

    for (std::size_t i = 0; i < B; ++i) {
        OscBlockTaskArgs args = make_args(q, dpdt, i);
        Legion::TaskLauncher launcher(
            OSC_CHAIN_BLOCK_TASK_ID,
            Legion::TaskArgument(&args, sizeof(OscBlockTaskArgs)));
        done.emplace_back(runtime->execute_task(ctx, launcher));
    }

    for (auto& f : done) f.get_void_result();
}

inline void osc_chain_gb(state_type& q, state_type& dpdt)
{
    // osc_chain already waits for completion (global barrier semantics)
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p)
{
    using checked_math::pow;
    using std::abs;
    const std::size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i + 1 < N; ++i) {
        e += 0.5 * p[i] * p[i]
          +  pow(q[i], KAPPA) / KAPPA
          +  pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
      +  pow(q[N - 1], KAPPA) / KAPPA
      +  0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

inline void append_block(dvec& out, const dvec& in)
{
    out.insert(out.end(), in.begin(), in.end());
}

inline void append_block(dvec& out, const shared_vec& in)
{
    if (!in) return;
    out.insert(out.end(), in->begin(), in->end());
}

template <typename S>
double energy(const S& q_blocks, const S& p_blocks)
{
    dvec q, p;
    for (std::size_t i = 0; i < q_blocks.size(); ++i) {
        append_block(q, q_blocks[i]);
        append_block(p, p_blocks[i]);
    }
    return energy(q, p);
}

#endif
