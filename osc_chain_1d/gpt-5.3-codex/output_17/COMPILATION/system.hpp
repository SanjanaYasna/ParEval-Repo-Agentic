// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow model to Legion task/future model
#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <boost/math/special_functions/sign.hpp>
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
    using boost::math::sign;
    return checked_math::pow(x, k) * sign(x);
}

typedef std::vector<double> dvec;
typedef std::vector<Legion::Future> state_type;

namespace legion_osc_detail {

enum : Legion::TaskID {
    SYSTEM_COMPUTE_BLOCK_TASK_ID = 32001
};

enum BlockKind : int {
    BLOCK_SINGLE = 0,
    BLOCK_FIRST  = 1,
    BLOCK_CENTER = 2,
    BLOCK_LAST   = 3
};

struct ComputeBlockArgs {
    int kind;
};

inline std::size_t future_len(const Legion::Future& f) {
    return f.get_size() / sizeof(double);
}

inline const double* future_data(const Legion::Future& f) {
    return static_cast<const double*>(f.get_buffer(Legion::Memory::SYSTEM_MEM));
}

inline double first_of(const Legion::Future& f) {
    const std::size_t n = future_len(f);
    if (n == 0) return 0.0;
    return future_data(f)[0];
}

inline double last_of(const Legion::Future& f) {
    const std::size_t n = future_len(f);
    if (n == 0) return 0.0;
    return future_data(f)[n - 1];
}

inline void compute_block_task(const Legion::Task* task,
                               const std::vector<Legion::PhysicalRegion>&,
                               Legion::Context ctx,
                               Legion::Runtime* runtime) {
    const ComputeBlockArgs* args =
        static_cast<const ComputeBlockArgs*>(task->args);
    const BlockKind kind = static_cast<BlockKind>(args->kind);

    const Legion::Future& qf = task->futures[0];
    const std::size_t N = future_len(qf);
    const double* q = (N > 0) ? future_data(qf) : nullptr;

    dvec dpdt(N, 0.0);

    if (N > 0) {
        if (kind == BLOCK_SINGLE || kind == BLOCK_FIRST) {
            const double q_r = (kind == BLOCK_SINGLE) ? 0.0 : first_of(task->futures[1]);

            double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
            for (std::size_t i = 0; i < N - 1; ++i) {
                dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
                coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
                dpdt[i] -= coupling_lr;
            }
            dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                        + coupling_lr
                        - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
        } else if (kind == BLOCK_CENTER) {
            const double q_l = last_of(task->futures[1]);
            const double q_r = first_of(task->futures[2]);

            double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
            for (std::size_t i = 0; i < N - 1; ++i) {
                dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
                coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
                dpdt[i] -= coupling_lr;
            }
            dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                        + coupling_lr
                        - signed_pow(q[N - 1] - q_r, LAMBDA - 1.0);
        } else {  // BLOCK_LAST
            const double q_l = last_of(task->futures[1]);

            double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1.0);
            for (std::size_t i = 0; i < N - 1; ++i) {
                dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
                coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
                dpdt[i] -= coupling_lr;
            }
            dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                        + coupling_lr
                        - signed_pow(q[N - 1], LAMBDA - 1.0);
        }
    }

    const void* out_ptr = dpdt.empty() ? nullptr : static_cast<const void*>(dpdt.data());
    Legion::Runtime::legion_task_postamble(
        runtime, ctx, out_ptr, dpdt.size() * sizeof(double));
}

inline Legion::Future launch_compute_block(Legion::Runtime* runtime,
                                           Legion::Context ctx,
                                           BlockKind kind,
                                           const Legion::Future& q,
                                           const Legion::Future* q_l = nullptr,
                                           const Legion::Future* q_r = nullptr) {
    ComputeBlockArgs args{static_cast<int>(kind)};
    Legion::TaskLauncher launcher(
        SYSTEM_COMPUTE_BLOCK_TASK_ID, Legion::TaskArgument(&args, sizeof(args)));
    launcher.add_future(q);

    if (kind == BLOCK_FIRST) {
        launcher.add_future(*q_r);
    } else if (kind == BLOCK_CENTER) {
        launcher.add_future(*q_l);
        launcher.add_future(*q_r);
    } else if (kind == BLOCK_LAST) {
        launcher.add_future(*q_l);
    }
    return runtime->execute_task(ctx, launcher);
}

inline bool g_tasks_registered = []() {
    Legion::TaskVariantRegistrar registrar(
        SYSTEM_COMPUTE_BLOCK_TASK_ID, "system_compute_block");
    registrar.add_constraint(
        Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<compute_block_task>(
        registrar, "system_compute_block");
    return true;
}();

inline thread_local Legion::Runtime* g_runtime = nullptr;
inline thread_local Legion::Context g_context = Legion::Context();
inline thread_local bool g_bound = false;

}  // namespace legion_osc_detail

inline void bind_legion_system_context(Legion::Runtime* runtime,
                                       Legion::Context context) {
    legion_osc_detail::g_runtime = runtime;
    legion_osc_detail::g_context = context;
    legion_osc_detail::g_bound = true;
}

inline void osc_chain(state_type& q, state_type& dpdt) {
    using namespace legion_osc_detail;
    (void)g_tasks_registered;

    if (!g_bound || g_runtime == nullptr) {
        throw std::runtime_error(
            "Legion system context not bound. Call bind_legion_system_context(...) "
            "from a running Legion task before osc_chain.");
    }

    const std::size_t N = q.size();
    if (dpdt.size() != N) dpdt.resize(N);
    if (N == 0) return;

    state_type next(N);

    if (N == 1) {
        next[0] = launch_compute_block(g_runtime, g_context, BLOCK_SINGLE, q[0]);
    } else {
        next[0] = launch_compute_block(g_runtime, g_context, BLOCK_FIRST, q[0], nullptr, &q[1]);
        for (std::size_t i = 1; i < N - 1; ++i) {
            next[i] = launch_compute_block(g_runtime, g_context, BLOCK_CENTER, q[i], &q[i - 1], &q[i + 1]);
        }
        next[N - 1] = launch_compute_block(g_runtime, g_context, BLOCK_LAST, q[N - 1], &q[N - 2], nullptr);
    }

    dpdt.swap(next);
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
    osc_chain(q, dpdt);
    for (auto& f : dpdt) {
        (void)f.get_buffer(Legion::Memory::SYSTEM_MEM); // barrier
    }
}

inline double energy(const dvec& q, const dvec& p) {
    using std::abs;
    const std::size_t N = q.size();
    double e = 0.5 * checked_math::pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i]
           + checked_math::pow(q[i], KAPPA) / KAPPA
           + checked_math::pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + checked_math::pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * checked_math::pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

template <typename S>
double energy(const S& q_fut, const S& p_fut) {
    dvec q, p;
    for (std::size_t i = 0; i < q_fut.size(); ++i) {
        const std::size_t qn = q_fut[i].get_size() / sizeof(double);
        const std::size_t pn = p_fut[i].get_size() / sizeof(double);

        const double* qb = (qn > 0)
            ? static_cast<const double*>(q_fut[i].get_buffer(Legion::Memory::SYSTEM_MEM))
            : nullptr;
        const double* pb = (pn > 0)
            ? static_cast<const double*>(p_fut[i].get_buffer(Legion::Memory::SYSTEM_MEM))
            : nullptr;

        q.insert(q.end(), qb, qb + qn);
        p.insert(p.end(), pb, pb + pn);
    }
    return energy(q, p);
}

#endif // SYSTEM_HPP
