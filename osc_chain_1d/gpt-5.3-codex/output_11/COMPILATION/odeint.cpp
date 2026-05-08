// Translated from HPX to Legion execution model.
//
// This version keeps the same command-line interface:
//   --N, --G, --steps, --dt
// and writes results to odeint.txt.
//
// Example:
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 4

#include "legion.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace Legion;

namespace {

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  COMPUTE_FORCE_TASK_ID,
  UPDATE_P_TASK_ID,
  UPDATE_Q_TASK_ID
};

enum FieldIDs {
  FID_Q = 1,
  FID_P,
  FID_DPDT
};

struct SimConfig {
  std::size_t N = 1024;
  std::size_t G = 128;
  std::size_t steps = 100;
  double dt = 0.01;
  bool help = false;
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  return checked_pow(x, k) * s;
}

inline bool starts_with(const std::string &s, const std::string &pfx) {
  return s.size() >= pfx.size() && s.compare(0, pfx.size(), pfx) == 0;
}

inline bool parse_size_t(const std::string &s, std::size_t &out) {
  try {
    out = static_cast<std::size_t>(std::stoull(s));
    return true;
  } catch (...) {
    return false;
  }
}

inline bool parse_double(const std::string &s, double &out) {
  try {
    out = std::stod(s);
    return true;
  } catch (...) {
    return false;
  }
}

SimConfig parse_command_line(const InputArgs &args) {
  SimConfig cfg;
  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);

    if (a == "--help" || a == "-h") {
      cfg.help = true;
      continue;
    }

    auto parse_keyval = [&](const std::string &key, auto &&setter) {
      if (starts_with(a, key + "=")) {
        setter(a.substr(key.size() + 1));
        return true;
      }
      if (a == key && i + 1 < args.argc) {
        setter(std::string(args.argv[++i]));
        return true;
      }
      return false;
    };

    if (parse_keyval("--N", [&](const std::string &v) {
          std::size_t x;
          if (parse_size_t(v, x)) cfg.N = x;
        }))
      continue;

    if (parse_keyval("--G", [&](const std::string &v) {
          std::size_t x;
          if (parse_size_t(v, x)) cfg.G = x;
        }))
      continue;

    if (parse_keyval("--steps", [&](const std::string &v) {
          std::size_t x;
          if (parse_size_t(v, x)) cfg.steps = x;
        }))
      continue;

    if (parse_keyval("--dt", [&](const std::string &v) {
          double x;
          if (parse_double(v, x)) cfg.dt = x;
        }))
      continue;
  }

  if (cfg.N == 0) cfg.N = 1;
  if (cfg.G == 0) cfg.G = 1;
  return cfg;
}

double compute_total_energy(Context ctx, Runtime *runtime, LogicalRegion lr_state) {
  RegionRequirement req(lr_state, READ_ONLY, EXCLUSIVE, lr_state);
  req.add_field(FID_Q);
  req.add_field(FID_P);

  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_ONLY, double, 1> q_acc(pr, FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(pr, FID_P);

  const Rect<1> rect = runtime->get_index_space_domain(ctx, lr_state.get_index_space());
  const coord_t lo = rect.lo[0];
  const coord_t hi = rect.hi[0];

  double energy = 0.0;
  if (hi >= lo) {
    energy = 0.5 * checked_pow(std::abs(q_acc[Point<1>(lo)]), LAMBDA) / LAMBDA;

    for (coord_t i = lo; i < hi; ++i) {
      const double qi = q_acc[Point<1>(i)];
      const double qip1 = q_acc[Point<1>(i + 1)];
      const double pi = p_acc[Point<1>(i)];
      energy += 0.5 * pi * pi
              + checked_pow(qi, KAPPA) / KAPPA
              + checked_pow(std::abs(qi - qip1), LAMBDA) / LAMBDA;
    }

    const double qn = q_acc[Point<1>(hi)];
    const double pn = p_acc[Point<1>(hi)];
    energy += 0.5 * pn * pn
            + checked_pow(qn, KAPPA) / KAPPA
            + 0.5 * checked_pow(std::abs(qn), LAMBDA) / LAMBDA;
  }

  runtime->unmap_region(ctx, pr);
  return energy;
}

void compute_force_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime) {
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[0], FID_DPDT);
  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[1], FID_Q);

  const Rect<1> subrect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  const Rect<1> fullrect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  const coord_t glo = fullrect.lo[0];
  const coord_t ghi = fullrect.hi[0];

  for (PointInRectIterator<1> pir(subrect); pir(); pir++) {
    const Point<1> p = *pir;
    const coord_t i = p[0];
    const double qi = q_acc[p];
    const double ql = (i == glo) ? 0.0 : q_acc[Point<1>(i - 1)];
    const double qr = (i == ghi) ? 0.0 : q_acc[Point<1>(i + 1)];

    const double force =
        -signed_pow(qi, KAPPA - 1.0)
        + signed_pow(ql - qi, LAMBDA - 1.0)
        - signed_pow(qi - qr, LAMBDA - 1.0);

    dpdt_acc[p] = force;
  }
}

void update_p_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  double scale_dt = 0.0;
  if (task->arglen == sizeof(double)) {
    std::memcpy(&scale_dt, task->args, sizeof(double));
  }

  FieldAccessor<READ_WRITE, double, 1> p_acc(regions[0], FID_P);
  FieldAccessor<READ_ONLY, double, 1> dpdt_acc(regions[1], FID_DPDT);

  const Rect<1> subrect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (PointInRectIterator<1> pir(subrect); pir(); pir++) {
    const Point<1> p = *pir;
    p_acc[p] = p_acc[p] + scale_dt * dpdt_acc[p];
  }
}

void update_q_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  double scale_dt = 0.0;
  if (task->arglen == sizeof(double)) {
    std::memcpy(&scale_dt, task->args, sizeof(double));
  }

  FieldAccessor<READ_WRITE, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_P);

  const Rect<1> subrect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (PointInRectIterator<1> pir(subrect); pir(); pir++) {
    const Point<1> p = *pir;
    q_acc[p] = q_acc[p] + scale_dt * p_acc[p];
  }
}

void launch_compute_force(Context ctx, Runtime *runtime,
                          Domain color_domain,
                          LogicalPartition lp_blocks,
                          LogicalRegion lr_state) {
  IndexLauncher launcher(COMPUTE_FORCE_TASK_ID, color_domain, TaskArgument(nullptr, 0), ArgumentMap());

  launcher.add_region_requirement(
      RegionRequirement(lp_blocks, 0 /*identity projection*/, WRITE_DISCARD, EXCLUSIVE, lr_state));
  launcher.region_requirements[0].add_field(FID_DPDT);

  launcher.add_region_requirement(
      RegionRequirement(lr_state, READ_ONLY, EXCLUSIVE, lr_state));
  launcher.region_requirements[1].add_field(FID_Q);

  runtime->execute_index_space(ctx, launcher);
}

void launch_update_p(Context ctx, Runtime *runtime,
                     Domain color_domain,
                     LogicalPartition lp_blocks,
                     LogicalRegion lr_state,
                     double scale_dt) {
  IndexLauncher launcher(UPDATE_P_TASK_ID, color_domain,
                         TaskArgument(&scale_dt, sizeof(double)), ArgumentMap());

  launcher.add_region_requirement(
      RegionRequirement(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr_state));
  launcher.region_requirements[0].add_field(FID_P);

  launcher.add_region_requirement(
      RegionRequirement(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr_state));
  launcher.region_requirements[1].add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

void launch_update_q(Context ctx, Runtime *runtime,
                     Domain color_domain,
                     LogicalPartition lp_blocks,
                     LogicalRegion lr_state,
                     double scale_dt) {
  IndexLauncher launcher(UPDATE_Q_TASK_ID, color_domain,
                         TaskArgument(&scale_dt, sizeof(double)), ArgumentMap());

  launcher.add_region_requirement(
      RegionRequirement(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr_state));
  launcher.region_requirements[0].add_field(FID_Q);

  launcher.add_region_requirement(
      RegionRequirement(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr_state));
  launcher.region_requirements[1].add_field(FID_P);

  runtime->execute_index_space(ctx, launcher);
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
  const SimConfig cfg = parse_command_line(Runtime::get_input_args());

  if (cfg.help) {
    std::cout << "Usage: ./odeint [options]\n"
                 "  --N <int>      Dimension (default 1024)\n"
                 "  --G <int>      Block size (default 128)\n"
                 "  --steps <int>  Number of time steps (default 100)\n"
                 "  --dt <double>  Step size (default 0.01)\n";
    return;
  }

  const std::size_t M = (cfg.N + cfg.G - 1) / cfg.G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << cfg.N
          << ", number of elements per dataflow: " << cfg.G
          << ", number of dataflow: " << M
          << ", steps: " << cfg.steps
          << ", dt: " << cfg.dt << "\n";

  // Create state region: q, p, dpdt over 1D domain [0, N-1].
  Rect<1> elem_bounds(0, static_cast<coord_t>(cfg.N - 1));
  IndexSpace is_elements = runtime->create_index_space(ctx, elem_bounds);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
    alloc.allocate_field(sizeof(double), FID_Q);
    alloc.allocate_field(sizeof(double), FID_P);
    alloc.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr_state = runtime->create_logical_region(ctx, is_elements, fs);

  // Partition elements into M blocks (for parallel index launches).
  Rect<1> color_bounds(0, static_cast<coord_t>(M - 1));
  IndexSpace is_colors = runtime->create_index_space(ctx, color_bounds);
  IndexPartition ip_blocks = runtime->create_equal_partition(ctx, is_elements, is_colors);
  LogicalPartition lp_blocks = runtime->get_logical_partition(ctx, lr_state, ip_blocks);
  Domain color_domain = runtime->get_index_space_domain(ctx, is_colors);

  // Initialize q=0, p=random in [-1,1], dpdt=0.
  {
    RegionRequirement req(lr_state, WRITE_DISCARD, EXCLUSIVE, lr_state);
    req.add_field(FID_Q);
    req.add_field(FID_P);
    req.add_field(FID_DPDT);

    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> q_acc(pr, FID_Q);
    FieldAccessor<WRITE_DISCARD, double, 1> p_acc(pr, FID_P);
    FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(pr, FID_DPDT);

    std::mt19937 engine(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (coord_t i = elem_bounds.lo[0]; i <= elem_bounds.hi[0]; ++i) {
      Point<1> p(i);
      q_acc[p] = 0.0;
      p_acc[p] = dist(engine);
      dpdt_acc[p] = 0.0;
    }

    runtime->unmap_region(ctx, pr);
  }

  const double e0 = compute_total_energy(ctx, runtime, lr_state);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(e0)) << "\n";

  // Time integration (symplectic velocity-Verlet):
  // p_{n+1/2} = p_n + (dt/2) * a(q_n)
  // q_{n+1}   = q_n + dt * p_{n+1/2}
  // p_{n+1}   = p_{n+1/2} + (dt/2) * a(q_{n+1})
  for (std::size_t step = 0; step < cfg.steps; ++step) {
    launch_compute_force(ctx, runtime, color_domain, lp_blocks, lr_state);
    launch_update_p(ctx, runtime, color_domain, lp_blocks, lr_state, 0.5 * cfg.dt);
    launch_update_q(ctx, runtime, color_domain, lp_blocks, lr_state, cfg.dt);
    launch_compute_force(ctx, runtime, color_domain, lp_blocks, lr_state);
    launch_update_p(ctx, runtime, color_domain, lp_blocks, lr_state, 0.5 * cfg.dt);
  }

  runtime->issue_execution_fence(ctx);

  const double e1 = compute_total_energy(ctx, runtime, lr_state);
  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(e1)) << "\n";

  // Cleanup Legion objects.
  runtime->destroy_logical_region(ctx, lr_state);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip_blocks);
  runtime->destroy_index_space(ctx, is_colors);
  runtime->destroy_index_space(ctx, is_elements);
}

} // namespace

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(COMPUTE_FORCE_TASK_ID, "compute_force");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<compute_force_task>(registrar, "compute_force");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_P_TASK_ID, "update_p");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<update_p_task>(registrar, "update_p");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_Q_TASK_ID, "update_q");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<update_q_task>(registrar, "update_q");
  }

  return Runtime::start(argc, argv);
}
