// Copyright 2013 Mario Mulansky
// Translated to Legion execution model (default mapper)

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
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
    if (x == 0.0) return 0.0;
    return checked_math::pow(x, k) * (x > 0.0 ? 1.0 : -1.0);
}

using dvec = std::vector<double>;
using shared_vec = std::shared_ptr<dvec>;  // kept for compatibility with other headers

enum : Legion::FieldID {
    FID_VALUE = 1
};

enum : Legion::TaskID {
    OSC_BLOCK_TASK_ID = 10001
};

struct state_type {
    std::size_t N{0};
    std::size_t G{0};
    std::size_t M{0};

    Legion::IndexSpace is;
    Legion::FieldSpace fs;
    Legion::LogicalRegion lr;

    Legion::IndexPartition ip_blocks;
    Legion::LogicalPartition lp_blocks;

    bool initialized{false};
};

inline Legion::Runtime*& bound_runtime() {
    static Legion::Runtime* rt = nullptr;
    return rt;
}

inline Legion::Context& bound_context() {
    static Legion::Context ctx = Legion::Context::NO_CONTEXT;
    return ctx;
}

inline void bind_legion_context(Legion::Runtime* runtime, Legion::Context ctx) {
    bound_runtime() = runtime;
    bound_context() = ctx;
}

inline state_type make_state(Legion::Runtime* runtime, Legion::Context ctx, std::size_t N, std::size_t G) {
    if (N == 0 || G == 0) {
        throw std::invalid_argument("make_state: N and G must be > 0");
    }

    state_type s;
    s.N = N;
    s.G = G;
    s.M = (N + G - 1) / G;

    s.is = runtime->create_index_space(ctx, Legion::Rect<1>(0, static_cast<Legion::coord_t>(N - 1)));
    s.fs = runtime->create_field_space(ctx);

    {
        Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, s.fs);
        allocator.allocate_field(sizeof(double), FID_VALUE);
    }

    s.lr = runtime->create_logical_region(ctx, s.is, s.fs);

    Legion::DomainColoring coloring;
    for (std::size_t b = 0; b < s.M; ++b) {
        const Legion::coord_t lo = static_cast<Legion::coord_t>(b * G);
        const Legion::coord_t hi = static_cast<Legion::coord_t>(std::min((b + 1) * G, N) - 1);
        coloring[Legion::DomainPoint::from_point<1>(Legion::Point<1>(static_cast<Legion::coord_t>(b)))] =
            Legion::Domain(Legion::Rect<1>(lo, hi));
    }

    s.ip_blocks = runtime->create_index_partition(ctx, s.is, coloring, true /*disjoint*/);
    s.lp_blocks = runtime->get_logical_partition(ctx, s.lr, s.ip_blocks);

    s.initialized = true;
    return s;
}

inline void destroy_state(Legion::Runtime* runtime, Legion::Context ctx, state_type& s) {
    if (!s.initialized) return;
    runtime->destroy_logical_region(ctx, s.lr);
    runtime->destroy_field_space(ctx, s.fs);
    runtime->destroy_index_partition(ctx, s.ip_blocks);
    runtime->destroy_index_space(ctx, s.is);
    s = state_type{};
}

inline void fill_state(Legion::Runtime* runtime, Legion::Context ctx, const state_type& s, double value) {
    runtime->fill_field(ctx, s.lr, s.lr, FID_VALUE, &value, sizeof(value));
}

inline void copy_from_vector(Legion::Runtime* runtime, Legion::Context ctx, const state_type& s, const dvec& values) {
    if (values.size() != s.N) {
        throw std::invalid_argument("copy_from_vector: values.size() must equal state.N");
    }

    Legion::RegionRequirement req(s.lr, Legion::WRITE_DISCARD, Legion::EXCLUSIVE, s.lr);
    req.add_field(FID_VALUE);

    Legion::InlineLauncher launcher(req);
    Legion::PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> acc(pr, FID_VALUE);
    for (Legion::coord_t i = 0; i < static_cast<Legion::coord_t>(s.N); ++i) {
        acc[Legion::Point<1>(i)] = values[static_cast<std::size_t>(i)];
    }

    runtime->unmap_region(ctx, pr);
}

inline dvec to_vector(Legion::Runtime* runtime, Legion::Context ctx, const state_type& s) {
    dvec out(s.N);

    Legion::RegionRequirement req(s.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, s.lr);
    req.add_field(FID_VALUE);

    Legion::InlineLauncher launcher(req);
    Legion::PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    Legion::FieldAccessor<Legion::READ_ONLY, double, 1> acc(pr, FID_VALUE);
    for (Legion::coord_t i = 0; i < static_cast<Legion::coord_t>(s.N); ++i) {
        out[static_cast<std::size_t>(i)] = acc[Legion::Point<1>(i)];
    }

    runtime->unmap_region(ctx, pr);
    return out;
}

inline void osc_block_task(const Legion::Task* task,
                           const std::vector<Legion::PhysicalRegion>& regions,
                           Legion::Context ctx,
                           Legion::Runtime* runtime) {
    assert(regions.size() == 2);

    const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(regions[0], FID_VALUE);
    Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VALUE);

    const Legion::Rect<1> q_rect =
        runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space()).bounds<1>();
    const Legion::Rect<1> my_rect =
        runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space()).bounds<1>();

    const Legion::coord_t glo = q_rect.lo[0];
    const Legion::coord_t ghi = q_rect.hi[0];

    for (Legion::PointInRectIterator<1> pir(my_rect); pir(); ++pir) {
        const Legion::Point<1> p = *pir;
        const Legion::coord_t i = p[0];

        const double qi = q_acc[p];
        const double ql = (i == glo) ? 0.0 : q_acc[Legion::Point<1>(i - 1)];
        const double qr = (i == ghi) ? 0.0 : q_acc[Legion::Point<1>(i + 1)];

        dpdt_acc[p] = -signed_pow(qi, KAPPA - 1.0)
                      -signed_pow(qi - ql, LAMBDA - 1.0)
                      -signed_pow(qi - qr, LAMBDA - 1.0);
    }
}

inline void preregister_system_tasks() {
    static bool done = false;
    if (done) return;
    done = true;

    Legion::TaskVariantRegistrar registrar(OSC_BLOCK_TASK_ID, "osc_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<osc_block_task>(registrar, "osc_block_task");
}

inline Legion::FutureMap launch_osc_chain(Legion::Runtime* runtime,
                                          Legion::Context ctx,
                                          const state_type& q,
                                          state_type& dpdt) {
    assert(q.initialized && dpdt.initialized);
    assert(q.N == dpdt.N);

    const Legion::Domain color_space =
        runtime->get_index_partition_color_space(ctx, dpdt.ip_blocks);

    Legion::IndexLauncher launcher(
        OSC_BLOCK_TASK_ID, color_space, Legion::TaskArgument(nullptr, 0), Legion::ArgumentMap());

    Legion::RegionRequirement q_req(q.lr, Legion::READ_ONLY, Legion::SIMULTANEOUS, q.lr);
    q_req.add_field(FID_VALUE);
    launcher.add_region_requirement(q_req);

    Legion::RegionRequirement d_req(dpdt.lp_blocks, 0 /*identity projection*/,
                                    Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt.lr);
    d_req.add_field(FID_VALUE);
    launcher.add_region_requirement(d_req);

    return runtime->execute_index_space(ctx, launcher);
}

inline void osc_chain(Legion::Runtime* runtime, Legion::Context ctx, state_type& q, state_type& dpdt) {
    (void)launch_osc_chain(runtime, ctx, q, dpdt);  // async launch (no global barrier)
}

inline void osc_chain_gb(Legion::Runtime* runtime, Legion::Context ctx, state_type& q, state_type& dpdt) {
    Legion::FutureMap fm = launch_osc_chain(runtime, ctx, q, dpdt);
    fm.wait_all_results();  // global barrier behavior
}

// Backward-compatible signatures (require bind_legion_context() first)
inline void osc_chain(state_type& q, state_type& dpdt) {
    if (!bound_runtime()) throw std::runtime_error("osc_chain: Legion runtime/context not bound");
    osc_chain(bound_runtime(), bound_context(), q, dpdt);
}

inline void osc_chain_gb(state_type& q, state_type& dpdt) {
    if (!bound_runtime()) throw std::runtime_error("osc_chain_gb: Legion runtime/context not bound");
    osc_chain_gb(bound_runtime(), bound_context(), q, dpdt);
}

inline double energy(const dvec& q, const dvec& p) {
    using checked_math::pow;
    using std::abs;

    const std::size_t N = q.size();
    if (N == 0 || p.size() != N) return 0.0;

    double e = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i < N - 1; ++i) {
        e += 0.5 * p[i] * p[i] + pow(q[i], KAPPA) / KAPPA
             + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1] + pow(q[N - 1], KAPPA) / KAPPA
         + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

inline double energy(Legion::Runtime* runtime, Legion::Context ctx,
                     const state_type& q_state, const state_type& p_state) {
    return energy(to_vector(runtime, ctx, q_state), to_vector(runtime, ctx, p_state));
}

inline double energy(const state_type& q_state, const state_type& p_state) {
    if (!bound_runtime()) throw std::runtime_error("energy: Legion runtime/context not bound");
    return energy(bound_runtime(), bound_context(), q_state, p_state);
}

#endif
