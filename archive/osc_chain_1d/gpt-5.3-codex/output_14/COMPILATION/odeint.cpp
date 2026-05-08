// Translated from HPX to Legion execution model (default mapper).
// odeint.cpp

#include "legion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  COMPUTE_DPDT_TASK_ID,
  UPDATE_P_HALF_TASK_ID,
  UPDATE_Q_TASK_ID,
  ENERGY_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_Q = 1,
  FID_P,
  FID_DPDT
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct SimConfig {
  int64_t N;
  int64_t G;
  int64_t steps;
  double dt;
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x > 0.0) return checked_pow(x, k);
  if (x < 0.0) return -checked_pow(x, k);
  return 0.0;
}

static inline int64_t get_block_id(const Task *task) {
  return static_cast<int64_t>(task->index_point.point_data[0]);
}

static SimConfig parse_config(const InputArgs &args) {
  SimConfig cfg{1024, 128, 100, 0.01};

  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);
    try {
      if (a == "--N" && i + 1 < args.argc) cfg.N = std::stoll(args.argv[++i]);
      else if (a.rfind("--N=", 0) == 0) cfg.N = std::stoll(a.substr(4));
      else if (a == "--G" && i + 1 < args.argc) cfg.G = std::stoll(args.argv[++i]);
      else if (a.rfind("--G=", 0) == 0) cfg.G = std::stoll(a.substr(4));
      else if (a == "--steps" && i + 1 < args.argc) cfg.steps = std::stoll(args.argv[++i]);
      else if (a.rfind("--steps=", 0) == 0) cfg.steps = std::stoll(a.substr(8));
      else if (a == "--dt" && i + 1 < args.argc) cfg.dt = std::stod(args.argv[++i]);
      else if (a.rfind("--dt=", 0) == 0) cfg.dt = std::stod(a.substr(5));
    } catch (...) {
      // Ignore malformed user option values and keep defaults.
    }
  }

  if (cfg.N < 1) cfg.N = 1;
  if (cfg.G < 1) cfg.G = 1;
  if (cfg.steps < 0) cfg.steps = 0;
  return cfg;
}

void compute_dpdt_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context,
                       Runtime *) {
  const SimConfig &cfg = *reinterpret_cast<const SimConfig *>(task->args);
  const int64_t block = get_block_id(task);

  const int64_t start = block * cfg.G;
  const int64_t end = std::min<int64_t>(cfg.N, (block + 1) * cfg.G) - 1;
  if (start >= cfg.N || end < start) return;

  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[0], FID_DPDT);
  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[1], FID_Q);

  for (int64_t i = start; i <= end; ++i) {
    const Point<1> pi(i);
    const double qi = q_acc[pi];

    const double left = (i == 0)
                            ? -signed_pow(qi, LAMBDA - 1.0)
                            : -signed_pow(qi - q_acc[Point<1>(i - 1)], LAMBDA - 1.0);

    const double right = (i == cfg.N - 1)
                             ? -signed_pow(qi, LAMBDA - 1.0)
                             : -signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[pi] = -signed_pow(qi, KAPPA - 1.0) + left + right;
  }
}

void update_p_half_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context,
                        Runtime *) {
  const SimConfig &cfg = *reinterpret_cast<const SimConfig *>(task->args);
  const int64_t block = get_block_id(task);

  const int64_t start = block * cfg.G;
  const int64_t end = std::min<int64_t>(cfg.N, (block + 1) * cfg.G) - 1;
  if (start >= cfg.N || end < start) return;

  FieldAccessor<READ_WRITE, double, 1> p_acc(regions[0], FID_P);
  FieldAccessor<READ_ONLY, double, 1> dpdt_acc(regions[1], FID_DPDT);

  const double c = 0.5 * cfg.dt;
  for (int64_t i = start; i <= end; ++i) {
    const Point<1> pi(i);
    const double pval = p_acc[pi];
    p_acc[pi] = pval + c * dpdt_acc[pi];
  }
}

void update_q_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context,
                   Runtime *) {
  const SimConfig &cfg = *reinterpret_cast<const SimConfig *>(task->args);
  const int64_t block = get_block_id(task);

  const int64_t start = block * cfg.G;
  const int64_t end = std::min<int64_t>(cfg.N, (block + 1) * cfg.G) - 1;
  if (start >= cfg.N || end < start) return;

  FieldAccessor<READ_WRITE, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_P);

  for (int64_t i = start; i <= end; ++i) {
    const Point<1> pi(i);
    const double qval = q_acc[pi];
    q_acc[pi] = qval + cfg.dt * p_acc[pi];
  }
}

double energy_block_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context,
                         Runtime *) {
  const SimConfig &cfg = *reinterpret_cast<const SimConfig *>(task->args);
  const int64_t block = get_block_id(task);

  const int64_t start = block * cfg.G;
  const int64_t end = std::min<int64_t>(cfg.N, (block + 1) * cfg.G) - 1;
  if (start >= cfg.N || end < start) return 0.0;

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[0], FID_P);

  double e = 0.0;
  for (int64_t i = start; i <= end; ++i) {
    const double qi = q_acc[Point<1>(i)];
    const double pi = p_acc[Point<1>(i)];

    e += 0.5 * pi * pi + checked_pow(qi, KAPPA) / KAPPA;

    if (i == 0) e += 0.5 * checked_pow(qi, LAMBDA) / LAMBDA;
    if (i < cfg.N - 1) {
      const double qn = q_acc[Point<1>(i + 1)];
      e += checked_pow(qi - qn, LAMBDA) / LAMBDA;
    } else {
      e += 0.5 * checked_pow(qi, LAMBDA) / LAMBDA;
    }
  }

  return e;
}

static void launch_compute_dpdt(Context ctx, Runtime *runtime, const Domain &color_dom,
                                LogicalPartition lp_blocks, LogicalRegion lr,
                                const SimConfig &cfg) {
  ArgumentMap argmap;
  IndexLauncher launcher(COMPUTE_DPDT_TASK_ID, color_dom, TaskArgument(&cfg, sizeof(cfg)), argmap);

  RegionRequirement w_req(lp_blocks, 0 /*projection*/, WRITE_DISCARD, EXCLUSIVE, lr);
  w_req.add_field(FID_DPDT);
  launcher.add_region_requirement(w_req);

  RegionRequirement r_req(lr, READ_ONLY, EXCLUSIVE, lr);
  r_req.add_field(FID_Q);
  launcher.add_region_requirement(r_req);

  runtime->execute_index_space(ctx, launcher);
}

static void launch_update_p_half(Context ctx, Runtime *runtime, const Domain &color_dom,
                                 LogicalPartition lp_blocks, LogicalRegion lr,
                                 const SimConfig &cfg) {
  ArgumentMap argmap;
  IndexLauncher launcher(UPDATE_P_HALF_TASK_ID, color_dom, TaskArgument(&cfg, sizeof(cfg)), argmap);

  RegionRequirement p_req(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr);
  p_req.add_field(FID_P);
  launcher.add_region_requirement(p_req);

  RegionRequirement f_req(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr);
  f_req.add_field(FID_DPDT);
  launcher.add_region_requirement(f_req);

  runtime->execute_index_space(ctx, launcher);
}

static void launch_update_q(Context ctx, Runtime *runtime, const Domain &color_dom,
                            LogicalPartition lp_blocks, LogicalRegion lr,
                            const SimConfig &cfg) {
  ArgumentMap argmap;
  IndexLauncher launcher(UPDATE_Q_TASK_ID, color_dom, TaskArgument(&cfg, sizeof(cfg)), argmap);

  RegionRequirement q_req(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr);
  q_req.add_field(FID_Q);
  launcher.add_region_requirement(q_req);

  RegionRequirement p_req(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr);
  p_req.add_field(FID_P);
  launcher.add_region_requirement(p_req);

  runtime->execute_index_space(ctx, launcher);
}

static double compute_total_energy(Context ctx, Runtime *runtime, const Domain &color_dom,
                                   LogicalRegion lr, const SimConfig &cfg) {
  ArgumentMap argmap;
  IndexLauncher launcher(ENERGY_BLOCK_TASK_ID, color_dom, TaskArgument(&cfg, sizeof(cfg)), argmap);

  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(FID_Q);
  req.add_field(FID_P);
  launcher.add_region_requirement(req);

  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

  const int64_t num_blocks = (cfg.N + cfg.G - 1) / cfg.G;
  double total = 0.0;
  for (int64_t b = 0; b < num_blocks; ++b) {
    total += fm.get_result<double>(DomainPoint(Point<1>(b)));
  }
  return total;
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx,
                    Runtime *runtime) {
  SimConfig cfg = parse_config(Runtime::get_input_args());
  const int64_t num_blocks = (cfg.N + cfg.G - 1) / cfg.G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << cfg.N
          << ", number of elements per dataflow: " << cfg.G
          << ", number of dataflow: " << num_blocks
          << ", steps: " << cfg.steps
          << ", dt: " << cfg.dt << std::endl;

  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(Point<1>(0), Point<1>(cfg.N - 1)));
  IndexSpace color_is = runtime->create_index_space(
      ctx, Rect<1>(Point<1>(0), Point<1>(num_blocks - 1)));

  Transform<1, 1> transform;
  transform[0][0] = cfg.G;
  Rect<1> extent(Point<1>(0), Point<1>(cfg.G - 1));

  IndexPartition ip_blocks =
      runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);
  LogicalPartition lp_blocks;

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
    alloc.allocate_field(sizeof(double), FID_Q);
    alloc.allocate_field(sizeof(double), FID_P);
    alloc.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
  lp_blocks = runtime->get_logical_partition(ctx, lr, ip_blocks);

  // Initialize q=0, p=random(-1,1), dpdt=0
  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_Q);
    req.add_field(FID_P);
    req.add_field(FID_DPDT);

    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> q_acc(pr, FID_Q);
    FieldAccessor<WRITE_DISCARD, double, 1> p_acc(pr, FID_P);
    FieldAccessor<WRITE_DISCARD, double, 1> f_acc(pr, FID_DPDT);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::mt19937 engine(0);

    Rect<1> rect = runtime->get_index_space_domain(ctx, is).bounds<1, coord_t>();
    for (PointInRectIterator<1> it(rect); it(); it++) {
      q_acc[*it] = 0.0;
      p_acc[*it] = dist(engine);
      f_acc[*it] = 0.0;
    }

    runtime->unmap_region(ctx, pr);
  }

  Domain color_dom = runtime->get_index_space_domain(ctx, color_is);

  const double e_init = compute_total_energy(ctx, runtime, color_dom, lr, cfg);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(e_init)) << std::endl;

  // Integrate using a symplectic (velocity-Verlet style) scheme in Legion tasks.
  launch_compute_dpdt(ctx, runtime, color_dom, lp_blocks, lr, cfg);
  for (int64_t step = 0; step < cfg.steps; ++step) {
    launch_update_p_half(ctx, runtime, color_dom, lp_blocks, lr, cfg);
    launch_update_q(ctx, runtime, color_dom, lp_blocks, lr, cfg);
    launch_compute_dpdt(ctx, runtime, color_dom, lp_blocks, lr, cfg);
    launch_update_p_half(ctx, runtime, color_dom, lp_blocks, lr, cfg);
  }

  const double e_final = compute_total_energy(ctx, runtime, color_dom, lr, cfg);
  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(e_final)) << std::endl;

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip_blocks);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(COMPUTE_DPDT_TASK_ID, "compute_dpdt");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<compute_dpdt_task>(registrar, "compute_dpdt");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_P_HALF_TASK_ID, "update_p_half");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_p_half_task>(registrar, "update_p_half");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_Q_TASK_ID, "update_q");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_q_task>(registrar, "update_q");
  }

  {
    TaskVariantRegistrar registrar(ENERGY_BLOCK_TASK_ID, "energy_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<double, energy_block_task>(registrar, "energy_block");
  }

  return Runtime::start(argc, argv);
}
