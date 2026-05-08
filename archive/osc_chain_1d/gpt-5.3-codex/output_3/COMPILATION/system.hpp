// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion task launch model.

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <memory>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <stdexcept>

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
} // namespace checked_math

inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

typedef std::vector<double> dvec;
typedef std::shared_ptr<dvec> shared_vec;

// Legion translation uses concrete block storage (no HPX futures here).
typedef std::vector<shared_vec> state_type;

/*
 * Per-block compute kernels (same numerics as original HPX version).
 */
struct system_first_block
{
    shared_vec operator()(shared_vec q, const double q_r, shared_vec dpdt) const
    {
        const size_t N = q->size();
        double coupling_lr = -signed_pow((*q)[0], LAMBDA - 1);
        for (size_t i = 0; i < N - 1; ++i)
        {
            (*dpdt)[i] = -signed_pow((*q)[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((*q)[i] - (*q)[i + 1], LAMBDA - 1);
            (*dpdt)[i] -= coupling_lr;
        }
        (*dpdt)[N - 1] = -signed_pow((*q)[N - 1], KAPPA - 1)
                       + coupling_lr
                       - signed_pow((*q)[N - 1] - q_r, LAMBDA - 1);
        return dpdt;
    }
};

struct system_center_block
{
    shared_vec operator()(shared_vec q, const double q_l, const double q_r, shared_vec dpdt) const
    {
        const size_t N = q->size();
        double coupling_lr = -signed_pow((*q)[0] - q_l, LAMBDA - 1);
        for (size_t i = 0; i < N - 1; ++i)
        {
            (*dpdt)[i] = -signed_pow((*q)[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((*q)[i] - (*q)[i + 1], LAMBDA - 1);
            (*dpdt)[i] -= coupling_lr;
        }
        (*dpdt)[N - 1] = -signed_pow((*q)[N - 1], KAPPA - 1)
                       + coupling_lr
                       - signed_pow((*q)[N - 1] - q_r, LAMBDA - 1);
        return dpdt;
    }
};

struct system_last_block
{
    shared_vec operator()(shared_vec q, const double q_l, shared_vec dpdt) const
    {
        const size_t N = q->size();
        double coupling_lr = -signed_pow((*q)[0] - q_l, LAMBDA - 1);
        for (size_t i = 0; i < N - 1; ++i)
        {
            (*dpdt)[i] = -signed_pow((*q)[i], KAPPA - 1) + coupling_lr;
            coupling_lr = signed_pow((*q)[i] - (*q)[i + 1], LAMBDA - 1);
            (*dpdt)[i] -= coupling_lr;
        }
        (*dpdt)[N - 1] = -signed_pow((*q)[N - 1], KAPPA - 1)
                       + coupling_lr
                       - signed_pow((*q)[N - 1], LAMBDA - 1);
        return dpdt;
    }
};

/*
 * Legion task plumbing
 * --------------------
 * We launch one Legion task per block. Each task receives pointers to block data
 * (host memory) and computes dpdt for that block.
 *
 * NOTE: this is the closest structural translation of the HPX block-dataflow style
 * without introducing custom mapper logic or region redesign.
 */

enum : Legion::TaskID {
    OSC_BLOCK_TASK_ID = 10001
};

enum BlockKind : int {
    BLOCK_FIRST  = 0,
    BLOCK_CENTER = 1,
    BLOCK_LAST   = 2,
    BLOCK_SINGLE = 3
};

struct OscBlockTaskArgs {
    std::uintptr_t q_ptr_addr;
    std::uintptr_t dpdt_ptr_addr;
    double q_l;
    double q_r;
    int kind;
};

inline Legion::Context g_system_ctx = nullptr;
inline Legion::Runtime* g_system_runtime = nullptr;

inline void bind_system_runtime(Legion::Context ctx, Legion::Runtime* runtime)
{
    g_system_ctx = ctx;
    g_system_runtime = runtime;
}

inline void osc_block_task(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>&,
                           Legion::Context,
                           Legion::Runtime*)
{
    assert(task->arglen == sizeof(OscBlockTaskArgs));
    const auto* args = static_cast<const OscBlockTaskArgs*>(task->args);

    auto* q_ref_ptr = reinterpret_cast<const shared_vec*>(args->q_ptr_addr);
    auto* dpdt_ref_ptr = reinterpret_cast<shared_vec*>(args->dpdt_ptr_addr);

    if (q_ref_ptr == nullptr || dpdt_ref_ptr == nullptr || !(*q_ref_ptr) || !(*dpdt_ref_ptr))
        return;

    shared_vec q = *q_ref_ptr;
    shared_vec dpdt = *dpdt_ref_ptr;

    switch (args->kind)
    {
        case BLOCK_FIRST:
            system_first_block{}(q, args->q_r, dpdt);
            break;
        case BLOCK_CENTER:
            system_center_block{}(q, args->q_l, args->q_r, dpdt);
            break;
        case BLOCK_LAST:
            system_last_block{}(q, args->q_l, dpdt);
            break;
        case BLOCK_SINGLE:
        default:
            // Single block: use first-block form with right boundary = 0.
            system_first_block{}(q, 0.0, dpdt);
            break;
    }
}

inline void register_system_tasks()
{
    static bool registered = false;
    if (registered) return;
    registered = true;

    Legion::TaskVariantRegistrar registrar(OSC_BLOCK_TASK_ID, "osc_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<osc_block_task>(registrar, "osc_block_task");
}

/*
 * HPX equivalent: osc_chain
 * Legion version launches tasks and waits for completion for correctness with
 * in-memory block vectors.
 */
inline void osc_chain(const state_type& q, state_type& dpdt)
{
    if (g_system_runtime == nullptr || g_system_ctx == nullptr)
        throw std::runtime_error("bind_system_runtime(ctx, runtime) must be called before osc_chain().");

    const size_t N = q.size();
    dpdt.resize(N);

    std::vector<Legion::Future> done;
    done.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        if (!q[i]) throw std::runtime_error("q block is null.");
        if (!dpdt[i]) dpdt[i] = std::make_shared<dvec>(q[i]->size(), 0.0);
        else dpdt[i]->resize(q[i]->size());

        double q_l = 0.0, q_r = 0.0;
        if (i > 0 && q[i - 1] && !q[i - 1]->empty()) q_l = q[i - 1]->back();
        if (i + 1 < N && q[i + 1] && !q[i + 1]->empty()) q_r = q[i + 1]->front();

        int kind = BLOCK_CENTER;
        if (N == 1) kind = BLOCK_SINGLE;
        else if (i == 0) kind = BLOCK_FIRST;
        else if (i == N - 1) kind = BLOCK_LAST;

        OscBlockTaskArgs args{
            reinterpret_cast<std::uintptr_t>(&q[i]),
            reinterpret_cast<std::uintptr_t>(&dpdt[i]),
            q_l,
            q_r,
            kind
        };

        Legion::TaskLauncher launcher(OSC_BLOCK_TASK_ID,
                                      Legion::TaskArgument(&args, sizeof(args)));
        done.emplace_back(g_system_runtime->execute_task(g_system_ctx, launcher));
    }

    // Ensure dpdt is fully computed before returning.
    for (auto& f : done) f.get_void_result();
}

inline void osc_chain_gb(const state_type& q, state_type& dpdt)
{
    // With this Legion translation, osc_chain already enforces completion.
    osc_chain(q, dpdt);
}

inline double energy(const dvec& q, const dvec& p)
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i)
    {
        e += 0.5 * p[i] * p[i]
           + pow(q[i], KAPPA) / KAPPA
           + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

inline double energy(const state_type& q_blocks, const state_type& p_blocks)
{
    dvec q, p;
    for (size_t i = 0; i < q_blocks.size(); ++i)
    {
        q.insert(q.end(), q_blocks[i]->begin(), q_blocks[i]->end());
        p.insert(p.end(), p_blocks[i]->begin(), p_blocks[i]->end());
    }
    return energy(q, p);
}

#endif // SYSTEM_HPP
