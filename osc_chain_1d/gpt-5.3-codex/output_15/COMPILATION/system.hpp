// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion task-based execution model.

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <legion.h>

#include <vector>
#include <cmath>
#include <cstddef>
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

using dvec = std::vector<double>;

/*
 * Legion layout conventions used by this file:
 * - Each state lives in one LogicalRegion (1D index space, one double field).
 * - For block parallelism, dpdt uses a disjoint LogicalPartition into blocks.
 */
enum SystemTaskIDs : Legion::TaskID {
    OSC_CHAIN_BLOCK_TASK_ID = 10001
};

enum SystemFieldIDs : Legion::FieldID {
    FID_VALUE = 1
};

struct state_type {
    Legion::LogicalRegion lr = Legion::LogicalRegion::NO_REGION;
    Legion::LogicalPartition lp = Legion::LogicalPartition::NO_PART; // disjoint block partition
};

// Computes dpdt for one block (subregion of dpdt partition).
inline void osc_chain_block_task(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context ctx,
    Legion::Runtime* runtime)
{
    assert(task != nullptr);
    assert(regions.size() == 2);

    // regions[0] : full q region (READ_ONLY)
    // regions[1] : one block of dpdt (WRITE_DISCARD)
    const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(regions[0], FID_VALUE);
    Legion::FieldAccessor<Legion::WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VALUE);

    const Legion::Rect<1> q_rect =
        runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    const Legion::Rect<1> out_rect =
        runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

    const Legion::coord_t q_lo = q_rect.lo[0];
    const Legion::coord_t q_hi = q_rect.hi[0];

    for (Legion::PointInRectIterator<1> it(out_rect); it(); it++) {
        const Legion::coord_t idx = (*it)[0];
        const double qi = q_acc[*it];

        const double coupling_left =
            (idx == q_lo)
                ? -signed_pow(qi, LAMBDA - 1.0)
                : -signed_pow(qi - q_acc[Legion::Point<1>(idx - 1)], LAMBDA - 1.0);

        const double coupling_right =
            (idx == q_hi)
                ? signed_pow(qi, LAMBDA - 1.0)
                : signed_pow(qi - q_acc[Legion::Point<1>(idx + 1)], LAMBDA - 1.0);

        dpdt_acc[*it] = -signed_pow(qi, KAPPA - 1.0) + coupling_left - coupling_right;
    }
}

// Call once during startup (before Runtime::start).
inline void register_system_tasks() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    Legion::TaskVariantRegistrar registrar(OSC_CHAIN_BLOCK_TASK_ID, "osc_chain_block_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<osc_chain_block_task>(
        registrar, "osc_chain_block_task");
}

// Asynchronous launch (no global barrier).
inline Legion::FutureMap osc_chain(
    const state_type& q,
    state_type& dpdt,
    Legion::Runtime* runtime,
    Legion::Context ctx)
{
    assert(runtime != nullptr);
    assert(q.lr.exists());
    assert(dpdt.lr.exists());
    assert(dpdt.lp.exists());

    Legion::Domain launch_domain =
        runtime->get_index_partition_color_space(ctx, dpdt.lp.get_index_partition());

    Legion::ArgumentMap arg_map;
    Legion::IndexLauncher launcher(
        OSC_CHAIN_BLOCK_TASK_ID, launch_domain, Legion::TaskArgument(nullptr, 0), arg_map);

    // Full q region is read by every block task.
    launcher.add_region_requirement(
        Legion::RegionRequirement(q.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, q.lr));
    launcher.region_requirements.back().add_field(FID_VALUE);

    // Each block task writes only its projected dpdt subregion.
    launcher.add_region_requirement(
        Legion::RegionRequirement(dpdt.lp, 0 /*identity projection*/,
                                  Legion::WRITE_DISCARD, Legion::EXCLUSIVE, dpdt.lr));
    launcher.region_requirements.back().add_field(FID_VALUE);

    return runtime->execute_index_space(ctx, launcher);
}

// Synchronous variant with a global barrier.
inline void osc_chain_gb(
    const state_type& q,
    state_type& dpdt,
    Legion::Runtime* runtime,
    Legion::Context ctx)
{
    Legion::FutureMap fm = osc_chain(q, dpdt, runtime, ctx);
    fm.wait_all_results();
}

inline double energy(const dvec& q, const dvec& p) {
    using checked_math::pow;
    using std::abs;

    const std::size_t N = q.size();
    if (N == 0) return 0.0;

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

// Energy directly from Legion regions (host-side inline mapping).
inline double energy(
    const state_type& q_state,
    const state_type& p_state,
    Legion::Runtime* runtime,
    Legion::Context ctx)
{
    assert(runtime != nullptr);
    assert(q_state.lr.exists());
    assert(p_state.lr.exists());

    Legion::RegionRequirement q_req(
        q_state.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VALUE);
    Legion::PhysicalRegion q_pr = runtime->map_region(ctx, Legion::InlineLauncher(q_req));
    q_pr.wait_until_valid();

    Legion::RegionRequirement p_req(
        p_state.lr, Legion::READ_ONLY, Legion::EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VALUE);
    Legion::PhysicalRegion p_pr = runtime->map_region(ctx, Legion::InlineLauncher(p_req));
    p_pr.wait_until_valid();

    const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> q_acc(q_pr, FID_VALUE);
    const Legion::FieldAccessor<Legion::READ_ONLY, double, 1> p_acc(p_pr, FID_VALUE);

    const Legion::Rect<1> q_rect =
        runtime->get_index_space_domain(ctx, q_state.lr.get_index_space());
    const Legion::Rect<1> p_rect =
        runtime->get_index_space_domain(ctx, p_state.lr.get_index_space());
    assert(q_rect.lo[0] == p_rect.lo[0] && q_rect.hi[0] == p_rect.hi[0]);

    const Legion::coord_t lo = q_rect.lo[0];
    const Legion::coord_t hi = q_rect.hi[0];

    double e = 0.0;
    if (lo <= hi) {
        using checked_math::pow;
        using std::abs;

        e = 0.5 * pow(abs(q_acc[Legion::Point<1>(lo)]), LAMBDA) / LAMBDA;
        for (Legion::coord_t i = lo; i < hi; ++i) {
            const double qi = q_acc[Legion::Point<1>(i)];
            const double pi = p_acc[Legion::Point<1>(i)];
            const double qn = q_acc[Legion::Point<1>(i + 1)];
            e += 0.5 * pi * pi
               + pow(qi, KAPPA) / KAPPA
               + pow(abs(qi - qn), LAMBDA) / LAMBDA;
        }

        const double qlast = q_acc[Legion::Point<1>(hi)];
        const double plast = p_acc[Legion::Point<1>(hi)];
        e += 0.5 * plast * plast
           + pow(qlast, KAPPA) / KAPPA
           + 0.5 * pow(abs(qlast), LAMBDA) / LAMBDA;
    }

    runtime->unmap_region(ctx, p_pr);
    runtime->unmap_region(ctx, q_pr);
    return e;
}

#endif
