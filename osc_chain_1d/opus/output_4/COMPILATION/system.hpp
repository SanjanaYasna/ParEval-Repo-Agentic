// Copyright 2013 Mario Mulansky
// Translated from HPX execution model to Legion execution model

#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <vector>
#include <cmath>
#include <cassert>

#include "legion.h"
#include <boost/math/special_functions/sign.hpp>

using namespace Legion;

const double KAPPA = 3.5;
const double LAMBDA = 4.5;

// ---------- Field ID (shared across all translated files) ----------
enum {
    FID_VAL = 0,
};

// ---------- Task IDs for system tasks ----------
enum {
    FIRST_BLOCK_TASK_ID  = 20,
    CENTER_BLOCK_TASK_ID = 21,
    LAST_BLOCK_TASK_ID   = 22,
    SINGLE_BLOCK_TASK_ID = 23,
};

// ---------- State type for Legion ----------
typedef std::vector<double> dvec;

struct state_type {
    LogicalRegion    lr;       // full logical region (N elements)
    LogicalPartition lp;       // disjoint partition into M sub-regions
    size_t           M;        // number of blocks
    size_t           G;        // elements per block
    Context          ctx;
    Runtime         *runtime;

    size_t size() const { return M; }
};

// ---------- Checked math helpers ----------
namespace checked_math {
    inline double pow(double x, double y)
    {
        if (x == 0.0)
            return 0.0;
        using std::pow;
        using std::abs;
        return pow(abs(x), y);
    }
}

inline double signed_pow(double x, double k)
{
    using boost::math::sign;
    using std::abs;
    return checked_math::pow(x, k) * sign(x);
}

// ================================================================
// Pure computation kernels (no Legion dependency)
// ================================================================

inline void compute_first_block(const double *q, size_t N,
                                double q_r, double *dpdt)
{
    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_center_block(const double *q, size_t N,
                                 double q_l, double q_r, double *dpdt)
{
    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1] - q_r, LAMBDA - 1);
}

inline void compute_last_block(const double *q, size_t N,
                               double q_l, double *dpdt)
{
    double coupling_lr = -signed_pow(q[0] - q_l, LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);
}

inline void compute_single_block(const double *q, size_t N, double *dpdt)
{
    double coupling_lr = -signed_pow(q[0], LAMBDA - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        dpdt[i] = -signed_pow(q[i], KAPPA - 1) + coupling_lr;
        coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1);
        dpdt[i] -= coupling_lr;
    }
    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1)
        + coupling_lr - signed_pow(q[N - 1], LAMBDA - 1);
}

// ================================================================
// Legion task implementations
// ================================================================

// Helper: copy sub-region data into a local buffer via accessor
static inline void read_subregion(const FieldAccessor<READ_ONLY, double, 1> &acc,
                                  const Rect<1> &rect,
                                  std::vector<double> &buf)
{
    size_t N = rect.volume();
    buf.resize(N);
    for (coord_t j = rect.lo[0], idx = 0; j <= rect.hi[0]; ++j, ++idx)
        buf[idx] = acc[Point<1>(j)];
}

// Helper: write local buffer back into sub-region via accessor
static inline void write_subregion(const FieldAccessor<WRITE_DISCARD, double, 1> &acc,
                                   const Rect<1> &rect,
                                   const std::vector<double> &buf)
{
    for (coord_t j = rect.lo[0], idx = 0; j <= rect.hi[0]; ++j, ++idx)
        acc[Point<1>(j)] = buf[idx];
}

// ---- First block task ----
void first_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    acc_q_right(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[2], FID_VAL);

    Rect<1> rect_q = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect_right = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());

    std::vector<double> q_local, dpdt_local;
    read_subregion(acc_q, rect_q, q_local);
    size_t N = q_local.size();
    dpdt_local.resize(N);

    double q_r = acc_q_right[Point<1>(rect_right.lo[0])];

    compute_first_block(q_local.data(), N, q_r, dpdt_local.data());

    write_subregion(acc_dpdt, rect_q, dpdt_local);
}

// ---- Center block task ----
void center_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    acc_q_left(regions[1], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    acc_q_right(regions[2], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[3], FID_VAL);

    Rect<1> rect_q = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect_left = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());
    Rect<1> rect_right = runtime->get_index_space_domain(
        task->regions[2].region.get_index_space());

    std::vector<double> q_local, dpdt_local;
    read_subregion(acc_q, rect_q, q_local);
    size_t N = q_local.size();
    dpdt_local.resize(N);

    double q_l = acc_q_left[Point<1>(rect_left.hi[0])];
    double q_r = acc_q_right[Point<1>(rect_right.lo[0])];

    compute_center_block(q_local.data(), N, q_l, q_r, dpdt_local.data());

    write_subregion(acc_dpdt, rect_q, dpdt_local);
}

// ---- Last block task ----
void last_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1>    acc_q_left(regions[1], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[2], FID_VAL);

    Rect<1> rect_q = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());
    Rect<1> rect_left = runtime->get_index_space_domain(
        task->regions[1].region.get_index_space());

    std::vector<double> q_local, dpdt_local;
    read_subregion(acc_q, rect_q, q_local);
    size_t N = q_local.size();
    dpdt_local.resize(N);

    double q_l = acc_q_left[Point<1>(rect_left.hi[0])];

    compute_last_block(q_local.data(), N, q_l, dpdt_local.data());

    write_subregion(acc_dpdt, rect_q, dpdt_local);
}

// ---- Single block task ----
void single_block_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime)
{
    const FieldAccessor<READ_ONLY, double, 1>    acc_q(regions[0], FID_VAL);
    const FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[1], FID_VAL);

    Rect<1> rect_q = runtime->get_index_space_domain(
        task->regions[0].region.get_index_space());

    std::vector<double> q_local, dpdt_local;
    read_subregion(acc_q, rect_q, q_local);
    size_t N = q_local.size();
    dpdt_local.resize(N);

    compute_single_block(q_local.data(), N, dpdt_local.data());

    write_subregion(acc_dpdt, rect_q, dpdt_local);
}

// ================================================================
// osc_chain
// ================================================================
void osc_chain(state_type &q, state_type &dpdt)
{
    Context  ctx     = q.ctx;
    Runtime *runtime = q.runtime;
    const size_t M   = q.M;

    auto q_sub = [&](size_t i) {
        return runtime->get_logical_subregion_by_color(q.lp, DomainPoint(Point<1>((coord_t)i)));
    };
    auto dpdt_sub = [&](size_t i) {
        return runtime->get_logical_subregion_by_color(dpdt.lp, DomainPoint(Point<1>((coord_t)i)));
    };

    if (M == 1) {
        TaskLauncher launcher(SINGLE_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(0), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub(0), WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_field(1, FID_VAL);
        runtime->execute_task(ctx, launcher);
        return;
    }

    {
        TaskLauncher launcher(FIRST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(0), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(1), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub(0), WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_field(1, FID_VAL);
        launcher.add_field(2, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    for (size_t i = 1; i < M - 1; ++i) {
        TaskLauncher launcher(CENTER_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(i), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(i - 1), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(i + 1), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub(i), WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_field(1, FID_VAL);
        launcher.add_field(2, FID_VAL);
        launcher.add_field(3, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }

    {
        TaskLauncher launcher(LAST_BLOCK_TASK_ID, TaskArgument(NULL, 0));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(M - 1), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(q_sub(M - 2), READ_ONLY, EXCLUSIVE, q.lr));
        launcher.add_region_requirement(
            RegionRequirement(dpdt_sub(M - 1), WRITE_DISCARD, EXCLUSIVE, dpdt.lr));
        launcher.add_field(0, FID_VAL);
        launcher.add_field(1, FID_VAL);
        launcher.add_field(2, FID_VAL);
        runtime->execute_task(ctx, launcher);
    }
}

void osc_chain_gb(state_type &q, state_type &dpdt)
{
    osc_chain(q, dpdt);
    q.runtime->issue_execution_fence(q.ctx);
}

// ================================================================
// Energy computation
// ================================================================

double energy(const dvec &q, const dvec &p)
{
    using checked_math::pow;
    using std::abs;
    const size_t N = q.size();
    double en = 0.5 * pow(abs(q[0]), LAMBDA) / LAMBDA;
    for (size_t i = 0; i < N - 1; ++i) {
        en += 0.5 * p[i] * p[i]
            + pow(q[i], KAPPA) / KAPPA
            + pow(abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    en += 0.5 * p[N - 1] * p[N - 1]
        + pow(q[N - 1], KAPPA) / KAPPA
        + 0.5 * pow(abs(q[N - 1]), LAMBDA) / LAMBDA;
    return en;
}

double energy(const state_type &q_state, const state_type &p_state)
{
    Context  ctx     = q_state.ctx;
    Runtime *runtime = q_state.runtime;
    const size_t N   = q_state.M * q_state.G;

    RegionRequirement q_req(q_state.lr, READ_ONLY, EXCLUSIVE, q_state.lr);
    q_req.add_field(FID_VAL);
    InlineLauncher q_il(q_req);
    PhysicalRegion q_phys = runtime->map_region(ctx, q_il);
    q_phys.wait_until_valid();

    RegionRequirement p_req(p_state.lr, READ_ONLY, EXCLUSIVE, p_state.lr);
    p_req.add_field(FID_VAL);
    InlineLauncher p_il(p_req);
    PhysicalRegion p_phys = runtime->map_region(ctx, p_il);
    p_phys.wait_until_valid();

    const FieldAccessor<READ_ONLY, double, 1> acc_q(q_phys, FID_VAL);
    const FieldAccessor<READ_ONLY, double, 1> acc_p(p_phys, FID_VAL);

    Rect<1> rect = runtime->get_index_space_domain(
        q_state.lr.get_index_space());

    dvec q_vec(N), p_vec(N);
    for (coord_t j = rect.lo[0], idx = 0; j <= rect.hi[0]; ++j, ++idx) {
        q_vec[idx] = acc_q[Point<1>(j)];
        p_vec[idx] = acc_p[Point<1>(j)];
    }

    runtime->unmap_region(ctx, q_phys);
    runtime->unmap_region(ctx, p_phys);

    return energy(q_vec, p_vec);
}

// ================================================================
// Task registration
// ================================================================
inline void register_system_tasks()
{
    {
        TaskVariantRegistrar registrar(FIRST_BLOCK_TASK_ID, "first_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<first_block_task>(
            registrar, "first_block_task");
    }
    {
        TaskVariantRegistrar registrar(CENTER_BLOCK_TASK_ID, "center_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<center_block_task>(
            registrar, "center_block_task");
    }
    {
        TaskVariantRegistrar registrar(LAST_BLOCK_TASK_ID, "last_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<last_block_task>(
            registrar, "last_block_task");
    }
    {
        TaskVariantRegistrar registrar(SINGLE_BLOCK_TASK_ID, "single_block_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<single_block_task>(
            registrar, "single_block_task");
    }
}

#endif // SYSTEM_HPP
