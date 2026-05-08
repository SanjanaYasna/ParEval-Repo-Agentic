// Translated from HPX to Legion execution model (default mapper)
// odeint.cpp

#include "legion.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Legion;

namespace {

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs : TaskID {
  TOP_LEVEL_TASK_ID = 1,
  DERIVATIVE_TASK_ID,
  KICK_TASK_ID,
  DRIFT_TASK_ID,
  ENERGY_TASK_ID
};

enum FieldIDs : FieldID {
  FID_Q = 100,
  FID_P,
  FID_DPDT
};

struct ScalarArg {
  double value;
};

struct SimOptions {
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

inline double signum(double x) {
  return (x > 0.0) ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}

inline double signed_pow(double x, double k) {
  return checked_pow(x, k) * signum(x);
}

bool parse_size_t(const std::string &s, std::size_t &out) {
  try {
    std::size_t idx = 0;
    unsigned long long v = std::stoull(s, &idx);
    if (idx != s.size()) return false;
    out = static_cast<std::size_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string &s, double &out) {
  try {
    std::size_t idx = 0;
    double v = std::stod(s, &idx);
    if (idx != s.size()) return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}

bool parse_options(const InputArgs &args, SimOptions &opts) {
  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);

    if (a == "--help") {
      opts.help = true;
      continue;
    }

    auto parse_kv = [&](const std::string &name, std::string &value) -> bool {
      const std::string pref = "--" + name + "=";
      if (starts_with(a, pref)) {
        value = a.substr(pref.size());
        return true;
      }
      return false;
    };

    std::string value;
    if (a == "--N") {
      if (i + 1 >= args.argc || !parse_size_t(args.argv[++i], opts.N)) return false;
    } else if (parse_kv("N", value)) {
      if (!parse_size_t(value, opts.N)) return false;
    } else if (a == "--G") {
      if (i + 1 >= args.argc || !parse_size_t(args.argv[++i], opts.G)) return false;
    } else if (parse_kv("G", value)) {
      if (!parse_size_t(value, opts.G)) return false;
    } else if (a == "--steps") {
      if (i + 1 >= args.argc || !parse_size_t(args.argv[++i], opts.steps)) return false;
    } else if (parse_kv("steps", value)) {
      if (!parse_size_t(value, opts.steps)) return false;
    } else if (a == "--dt") {
      if (i + 1 >= args.argc || !parse_double(args.argv[++i], opts.dt)) return false;
    } else if (parse_kv("dt", value)) {
      if (!parse_double(value, opts.dt)) return false;
    } else {
      // Ignore unknown options (e.g. Legion runtime options like -ll:cpu)
    }
  }
  return true;
}

void print_usage() {
  std::cout
      << "Usage: ./odeint [options] [Legion runtime options]\n"
      << "Options:\n"
      << "  --N <int>       Dimension (default 1024)\n"
      << "  --G <int>       Block size (default 128)\n"
      << "  --steps <int>   Time steps (default 100)\n"
      << "  --dt <double>   Step size (default 0.01)\n"
      << "  --help          Show this message\n";
}

long long round_energy(double e) {
  if (!std::isfinite(e)) return 0LL;
  return static_cast<long long>(std::llround(e));
}

void derivative_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
  (void)task;
  const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_Q);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_DPDT);

  const Rect<1> full_rect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  const Rect<1> block_rect =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

  const coord_t lo = full_rect.lo[0];
  const coord_t hi = full_rect.hi[0];

  for (coord_t i = block_rect.lo[0]; i <= block_rect.hi[0]; ++i) {
    const double qi = q[Point<1>(i)];
    const double left =
        (i == lo) ? -signed_pow(qi, LAMBDA - 1.0)
                  : signed_pow(q[Point<1>(i - 1)] - qi, LAMBDA - 1.0);
    const double right =
        (i == hi) ? signed_pow(qi, LAMBDA - 1.0)
                  : signed_pow(qi - q[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt[Point<1>(i)] = -signed_pow(qi, KAPPA - 1.0) + left - right;
  }
}

void kick_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
  (void)ctx;
  (void)runtime;
  const auto *arg = static_cast<const ScalarArg *>(task->args);
  const double c = arg->value;

  FieldAccessor<READ_WRITE, double, 1> p(regions[0], FID_P);
  const FieldAccessor<READ_ONLY, double, 1> dpdt(regions[1], FID_DPDT);

  const Rect<1> block_rect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

  for (coord_t i = block_rect.lo[0]; i <= block_rect.hi[0]; ++i) {
    const Point<1> pt(i);
    p[pt] = p[pt] + c * dpdt[pt];
  }
}

void drift_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  (void)ctx;
  (void)runtime;
  const auto *arg = static_cast<const ScalarArg *>(task->args);
  const double dt = arg->value;

  FieldAccessor<READ_WRITE, double, 1> q(regions[0], FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_P);

  const Rect<1> block_rect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

  for (coord_t i = block_rect.lo[0]; i <= block_rect.hi[0]; ++i) {
    const Point<1> pt(i);
    q[pt] = q[pt] + dt * p[pt];
  }
}

double energy_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  (void)task;
  const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p(regions[0], FID_P);

  const Rect<1> r =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  const coord_t lo = r.lo[0];
  const coord_t hi = r.hi[0];

  double e = 0.5 * checked_pow(std::abs(q[Point<1>(lo)]), LAMBDA) / LAMBDA;

  for (coord_t i = lo; i < hi; ++i) {
    const double qi = q[Point<1>(i)];
    const double qip1 = q[Point<1>(i + 1)];
    const double pi = p[Point<1>(i)];

    e += 0.5 * pi * pi
         + checked_pow(qi, KAPPA) / KAPPA
         + checked_pow(std::abs(qi - qip1), LAMBDA) / LAMBDA;
  }

  const double qn = q[Point<1>(hi)];
  const double pn = p[Point<1>(hi)];
  e += 0.5 * pn * pn
       + checked_pow(qn, KAPPA) / KAPPA
       + 0.5 * checked_pow(std::abs(qn), LAMBDA) / LAMBDA;

  return e;
}

void launch_derivative(Runtime *runtime, Context ctx,
                       LogicalRegion lr, LogicalPartition lp_blocks,
                       const Domain &launch_domain) {
  IndexLauncher launcher(DERIVATIVE_TASK_ID, launch_domain, TaskArgument(nullptr, 0), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp_blocks, 0, WRITE_DISCARD, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

void launch_kick(Runtime *runtime, Context ctx,
                 LogicalRegion lr, LogicalPartition lp_blocks,
                 const Domain &launch_domain, double coeff_times_dt) {
  ScalarArg arg{coeff_times_dt};
  IndexLauncher launcher(KICK_TASK_ID, launch_domain, TaskArgument(&arg, sizeof(arg)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_P);

  launcher.add_region_requirement(RegionRequirement(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

void launch_drift(Runtime *runtime, Context ctx,
                  LogicalRegion lr, LogicalPartition lp_blocks,
                  const Domain &launch_domain, double dt) {
  ScalarArg arg{dt};
  IndexLauncher launcher(DRIFT_TASK_ID, launch_domain, TaskArgument(&arg, sizeof(arg)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp_blocks, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp_blocks, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_P);

  runtime->execute_index_space(ctx, launcher);
}

double launch_energy(Runtime *runtime, Context ctx, LogicalRegion lr) {
  TaskLauncher launcher(ENERGY_TASK_ID, TaskArgument(nullptr, 0));
  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_Q);
  launcher.region_requirements[0].add_field(FID_P);
  Future f = runtime->execute_task(ctx, launcher);
  return f.get_result<double>();
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  (void)task;
  (void)regions;

  SimOptions opts;
  InputArgs in = Runtime::get_input_args();
  if (!parse_options(in, opts)) {
    std::cerr << "Error: invalid command-line options.\n";
    print_usage();
    return;
  }
  if (opts.help) {
    print_usage();
    return;
  }

  if (opts.N == 0 || opts.G == 0 || opts.N < opts.G || (opts.N % opts.G) != 0) {
    std::cerr << "Error: require N > 0, G > 0, N >= G, and N % G == 0.\n";
    return;
  }

  const std::size_t N = opts.N;
  const std::size_t G = opts.G;
  const std::size_t M = N / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << opts.steps
          << ", dt: " << opts.dt << "\n";

  // Create Legion region: one element per oscillator index
  Rect<1> elem_rect(0, static_cast<coord_t>(N - 1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_P);
    allocator.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Partition into M blocks (each block size G)
  Rect<1> color_rect(0, static_cast<coord_t>(M - 1));
  IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
  IndexPartition ip_blocks = runtime->create_equal_partition(ctx, is, color_is);
  LogicalPartition lp_blocks = runtime->get_logical_partition(ctx, lr, ip_blocks);
  Domain launch_domain(color_rect);

  // Initialize q=0, p=random[-1,1], dpdt=0
  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_Q);
    req.add_field(FID_P);
    req.add_field(FID_DPDT);

    InlineLauncher init_launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, init_launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> q(pr, FID_Q);
    FieldAccessor<WRITE_DISCARD, double, 1> p(pr, FID_P);
    FieldAccessor<WRITE_DISCARD, double, 1> dpdt(pr, FID_DPDT);

    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);

    for (coord_t i = elem_rect.lo[0]; i <= elem_rect.hi[0]; ++i) {
      q[Point<1>(i)] = 0.0;
      p[Point<1>(i)] = distribution(engine);
      dpdt[Point<1>(i)] = 0.0;
    }

    runtime->unmap_region(ctx, pr);
  }

  const double e_init = launch_energy(runtime, ctx, lr);
  outfile << "Initialization complete, energy: " << round_energy(e_init) << "\n";

  // Symplectic RKN SB3A McLachlan integration
  if (opts.steps > 0) {
    constexpr double a1 = 0.40518861839525227722;
    constexpr double a2 = -0.28714404081652408900;
    constexpr double a3 = 0.5 - (a1 + a2);

    constexpr double b1 = -3.0 / 73.0;
    constexpr double b2 = 17.0 / 59.0;
    constexpr double b3 = 1.0 - 2.0 * (b1 + b2);

    const std::array<double, 6> a = {a1, a2, a3, a3, a2, a1};
    const std::array<double, 6> b = {b1, b2, b3, b2, b1, 0.0};

    for (std::size_t s = 0; s < opts.steps; ++s) {
      for (int stage = 0; stage < 6; ++stage) {
        const double drift_dt = a[stage] * opts.dt;
        if (drift_dt != 0.0) {
          launch_drift(runtime, ctx, lr, lp_blocks, launch_domain, drift_dt);
        }

        const double kick_dt = b[stage] * opts.dt;
        if (kick_dt != 0.0) {
          launch_derivative(runtime, ctx, lr, lp_blocks, launch_domain);
          launch_kick(runtime, ctx, lr, lp_blocks, launch_domain, kick_dt);
        }
      }
    }
  }

  const double e_final = launch_energy(runtime, ctx, lr);
  outfile << "Integration complete, energy: " << round_energy(e_final) << "\n";

  // Cleanup Legion objects
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_index_partition(ctx, ip_blocks);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
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
    TaskVariantRegistrar registrar(DERIVATIVE_TASK_ID, "derivative");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<derivative_task>(registrar, "derivative");
  }

  {
    TaskVariantRegistrar registrar(KICK_TASK_ID, "kick");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<kick_task>(registrar, "kick");
  }

  {
    TaskVariantRegistrar registrar(DRIFT_TASK_ID, "drift");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<drift_task>(registrar, "drift");
  }

  {
    TaskVariantRegistrar registrar(ENERGY_TASK_ID, "energy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<double, energy_task>(registrar, "energy");
  }

  return Runtime::start(argc, argv);
}
