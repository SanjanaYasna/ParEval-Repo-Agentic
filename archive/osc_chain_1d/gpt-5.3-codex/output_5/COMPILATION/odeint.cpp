// Translated from HPX to Legion execution model (default mapper)
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
  COMPUTE_FORCE_TASK_ID,
  UPDATE_P_TASK_ID,
  UPDATE_Q_TASK_ID,
  ENERGY_PARTIAL_TASK_ID
};

enum FieldIDs {
  FID_VAL = 1
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct BlockArgs {
  int64_t N;
  int64_t G;
  double alpha; // used by update tasks
};

struct Options {
  int64_t N = 1024;
  int64_t G = 128;
  int64_t steps = 100;
  double dt = 0.01;
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signum(double x) {
  return (x > 0.0) - (x < 0.0);
}

inline double signed_pow(double x, double k) {
  return checked_pow(x, k) * signum(x);
}

static bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}

static Options parse_options(const InputArgs &args) {
  Options opt;
  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);

    auto parse_next_i64 = [&](int64_t &dst) {
      if (i + 1 < args.argc) {
        dst = std::stoll(args.argv[++i]);
      }
    };
    auto parse_next_f64 = [&](double &dst) {
      if (i + 1 < args.argc) {
        dst = std::stod(args.argv[++i]);
      }
    };

    if (a == "--N") parse_next_i64(opt.N);
    else if (a == "--G") parse_next_i64(opt.G);
    else if (a == "--steps") parse_next_i64(opt.steps);
    else if (a == "--dt") parse_next_f64(opt.dt);
    else if (starts_with(a, "--N=")) opt.N = std::stoll(a.substr(4));
    else if (starts_with(a, "--G=")) opt.G = std::stoll(a.substr(4));
    else if (starts_with(a, "--steps=")) opt.steps = std::stoll(a.substr(8));
    else if (starts_with(a, "--dt=")) opt.dt = std::stod(a.substr(5));
    // ignore unknown args (including Legion runtime args)
  }

  if (opt.N < 1) opt.N = 1;
  if (opt.G < 1) opt.G = 1;
  if (opt.steps < 0) opt.steps = 0;
  if (opt.G > opt.N) opt.G = opt.N;
  return opt;
}

void compute_force_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context /*ctx*/, Runtime* /*runtime*/) {
  const BlockArgs &args = *reinterpret_cast<const BlockArgs*>(task->args);
  const int64_t block = static_cast<int64_t>(task->index_point[0]);

  const int64_t start = block * args.G;
  const int64_t end = std::min<int64_t>(args.N, start + args.G); // exclusive
  if (start >= end) return;

  FieldAccessor<WRITE_DISCARD, double, 1> acc_dpdt(regions[0], FID_VAL);
  FieldAccessor<READ_ONLY, double, 1> acc_q(regions[1], FID_VAL);

  for (int64_t i = start; i < end; ++i) {
    const Point<1> pi(static_cast<coord_t>(i));
    const double qi = acc_q[pi];

    const double left_coupling =
        (i == 0) ? -signed_pow(qi, LAMBDA - 1.0)
                 : signed_pow(acc_q[Point<1>(static_cast<coord_t>(i - 1))] - qi, LAMBDA - 1.0);

    const double right_coupling =
        (i == args.N - 1) ? signed_pow(qi, LAMBDA - 1.0)
                          : signed_pow(qi - acc_q[Point<1>(static_cast<coord_t>(i + 1))], LAMBDA - 1.0);

    acc_dpdt[pi] =
        -signed_pow(qi, KAPPA - 1.0) + left_coupling - right_coupling;
  }
}

void update_p_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context /*ctx*/, Runtime* /*runtime*/) {
  const BlockArgs &args = *reinterpret_cast<const BlockArgs*>(task->args);
  const int64_t block = static_cast<int64_t>(task->index_point[0]);

  const int64_t start = block * args.G;
  const int64_t end = std::min<int64_t>(args.N, start + args.G); // exclusive
  if (start >= end) return;

  FieldAccessor<READ_WRITE, double, 1> acc_p(regions[0], FID_VAL);
  FieldAccessor<READ_ONLY, double, 1> acc_dpdt(regions[1], FID_VAL);

  for (int64_t i = start; i < end; ++i) {
    const Point<1> pi(static_cast<coord_t>(i));
    const double updated = static_cast<double>(acc_p[pi]) + args.alpha * static_cast<double>(acc_dpdt[pi]);
    acc_p[pi] = updated;
  }
}

void update_q_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context /*ctx*/, Runtime* /*runtime*/) {
  const BlockArgs &args = *reinterpret_cast<const BlockArgs*>(task->args);
  const int64_t block = static_cast<int64_t>(task->index_point[0]);

  const int64_t start = block * args.G;
  const int64_t end = std::min<int64_t>(args.N, start + args.G); // exclusive
  if (start >= end) return;

  FieldAccessor<READ_WRITE, double, 1> acc_q(regions[0], FID_VAL);
  FieldAccessor<READ_ONLY, double, 1> acc_p(regions[1], FID_VAL);

  for (int64_t i = start; i < end; ++i) {
    const Point<1> pi(static_cast<coord_t>(i));
    const double updated = static_cast<double>(acc_q[pi]) + args.alpha * static_cast<double>(acc_p[pi]);
    acc_q[pi] = updated;
  }
}

double energy_partial_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context /*ctx*/, Runtime* /*runtime*/) {
  const BlockArgs &args = *reinterpret_cast<const BlockArgs*>(task->args);
  const int64_t block = static_cast<int64_t>(task->index_point[0]);

  const int64_t start = block * args.G;
  const int64_t end = std::min<int64_t>(args.N, start + args.G); // exclusive
  if (start >= end) return 0.0;

  FieldAccessor<READ_ONLY, double, 1> acc_q(regions[0], FID_VAL);
  FieldAccessor<READ_ONLY, double, 1> acc_p(regions[1], FID_VAL);

  double e = 0.0;
  for (int64_t i = start; i < end; ++i) {
    const double qi = acc_q[Point<1>(static_cast<coord_t>(i))];
    const double pi = acc_p[Point<1>(static_cast<coord_t>(i))];

    e += 0.5 * pi * pi + checked_pow(qi, KAPPA) / KAPPA;

    if (i < args.N - 1) {
      const double qnext = acc_q[Point<1>(static_cast<coord_t>(i + 1))];
      e += checked_pow(std::abs(qi - qnext), LAMBDA) / LAMBDA;
    }
    if (i == 0) {
      e += 0.5 * checked_pow(std::abs(qi), LAMBDA) / LAMBDA;
    }
    if (i == args.N - 1) {
      e += 0.5 * checked_pow(std::abs(qi), LAMBDA) / LAMBDA;
    }
  }

  return e;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime *runtime) {
  const Options opt = parse_options(Runtime::get_input_args());
  const int64_t N = opt.N;
  const int64_t G = opt.G;
  const int64_t steps = opt.steps;
  const double dt = opt.dt;
  const int64_t M = (N + G - 1) / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing." << std::endl;
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << steps
          << ", dt: " << dt << std::endl;

  // Create index/field/logical regions for q, p, dpdt
  const Rect<1> elem_rect(0, static_cast<coord_t>(N - 1));
  IndexSpace is = runtime->create_index_space(ctx, Domain(elem_rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_VAL);
  }

  LogicalRegion lr_q = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion lr_p = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion lr_dpdt = runtime->create_logical_region(ctx, is, fs);

  // Partition into fixed-size blocks via restriction
  const Rect<1> color_rect(0, static_cast<coord_t>(M - 1));
  IndexSpace color_is = runtime->create_index_space(ctx, Domain(color_rect));

  Transform<1, 1> transform;
  transform[0][0] = static_cast<coord_t>(G);
  const Rect<1> extent(0, static_cast<coord_t>(G - 1));

  IndexPartition ip_blocks =
      runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);

  LogicalPartition lp_q = runtime->get_logical_partition(ctx, lr_q, ip_blocks);
  LogicalPartition lp_p = runtime->get_logical_partition(ctx, lr_p, ip_blocks);
  LogicalPartition lp_dpdt = runtime->get_logical_partition(ctx, lr_dpdt, ip_blocks);

  // Initialize q = 0, dpdt = 0
  {
    const double zero = 0.0;
    runtime->fill_field(ctx, lr_q, lr_q, FID_VAL, &zero, sizeof(zero));
    runtime->fill_field(ctx, lr_dpdt, lr_dpdt, FID_VAL, &zero, sizeof(zero));
  }

  // Initialize p from uniform[-1, 1], seed 0
  {
    RegionRequirement req(lr_p, WRITE_DISCARD, EXCLUSIVE, lr_p);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> acc_p(pr, FID_VAL);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);

    for (int64_t i = 0; i < N; ++i) {
      acc_p[Point<1>(static_cast<coord_t>(i))] = distribution(engine);
    }
    runtime->unmap_region(ctx, pr);
  }

  auto launch_compute_force = [&](const BlockArgs &bargs) {
    IndexLauncher launcher(
        COMPUTE_FORCE_TASK_ID, Rect<1>(0, static_cast<coord_t>(M - 1)),
        TaskArgument(&bargs, sizeof(BlockArgs)), ArgumentMap());

    launcher.add_region_requirement(
        RegionRequirement(lp_dpdt, 0 /*proj*/, WRITE_DISCARD, EXCLUSIVE, lr_dpdt));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(lr_q, READ_ONLY, EXCLUSIVE, lr_q));
    launcher.region_requirements[1].add_field(FID_VAL);

    runtime->execute_index_space(ctx, launcher);
  };

  auto launch_update_p = [&](const BlockArgs &bargs) {
    IndexLauncher launcher(
        UPDATE_P_TASK_ID, Rect<1>(0, static_cast<coord_t>(M - 1)),
        TaskArgument(&bargs, sizeof(BlockArgs)), ArgumentMap());

    launcher.add_region_requirement(
        RegionRequirement(lp_p, 0 /*proj*/, READ_WRITE, EXCLUSIVE, lr_p));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(lp_dpdt, 0 /*proj*/, READ_ONLY, EXCLUSIVE, lr_dpdt));
    launcher.region_requirements[1].add_field(FID_VAL);

    runtime->execute_index_space(ctx, launcher);
  };

  auto launch_update_q = [&](const BlockArgs &bargs) {
    IndexLauncher launcher(
        UPDATE_Q_TASK_ID, Rect<1>(0, static_cast<coord_t>(M - 1)),
        TaskArgument(&bargs, sizeof(BlockArgs)), ArgumentMap());

    launcher.add_region_requirement(
        RegionRequirement(lp_q, 0 /*proj*/, READ_WRITE, EXCLUSIVE, lr_q));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(lp_p, 0 /*proj*/, READ_ONLY, EXCLUSIVE, lr_p));
    launcher.region_requirements[1].add_field(FID_VAL);

    runtime->execute_index_space(ctx, launcher);
  };

  auto compute_energy = [&](const BlockArgs &bargs) -> double {
    IndexLauncher launcher(
        ENERGY_PARTIAL_TASK_ID, Rect<1>(0, static_cast<coord_t>(M - 1)),
        TaskArgument(&bargs, sizeof(BlockArgs)), ArgumentMap());

    launcher.add_region_requirement(
        RegionRequirement(lr_q, READ_ONLY, EXCLUSIVE, lr_q));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(lr_p, READ_ONLY, EXCLUSIVE, lr_p));
    launcher.region_requirements[1].add_field(FID_VAL);

    FutureMap fmap = runtime->execute_index_space(ctx, launcher);

    double total = 0.0;
    for (int64_t b = 0; b < M; ++b) {
      DomainPoint dp(Point<1>(static_cast<coord_t>(b)));
      total += fmap.get_result<double>(dp);
    }
    return total;
  };

  BlockArgs common{N, G, 0.0};

  // Initial energy
  const double e_init = compute_energy(common);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(e_init)) << std::endl;

  // Velocity-Verlet integration (symplectic)
  launch_compute_force(common); // dpdt = F(q)

  for (int64_t s = 0; s < steps; ++s) {
    BlockArgs p_half{N, G, 0.5 * dt};
    launch_update_p(p_half);  // p += 0.5*dt*dpdt

    BlockArgs q_full{N, G, dt};
    launch_update_q(q_full);  // q += dt*p

    launch_compute_force(common); // dpdt = F(q_new)

    launch_update_p(p_half);  // p += 0.5*dt*dpdt_new
  }

  const double e_final = compute_energy(common);
  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(e_final)) << std::endl;

  // Ensure all work is complete before cleanup
  Future fence = runtime->issue_execution_fence(ctx);
  fence.wait();

  runtime->destroy_logical_region(ctx, lr_q);
  runtime->destroy_logical_region(ctx, lr_p);
  runtime->destroy_logical_region(ctx, lr_dpdt);
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
    TaskVariantRegistrar registrar(COMPUTE_FORCE_TASK_ID, "compute_force");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<compute_force_task>(registrar, "compute_force");
  }
  {
    TaskVariantRegistrar registrar(UPDATE_P_TASK_ID, "update_p");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_p_task>(registrar, "update_p");
  }
  {
    TaskVariantRegistrar registrar(UPDATE_Q_TASK_ID, "update_q");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_q_task>(registrar, "update_q");
  }
  {
    TaskVariantRegistrar registrar(ENERGY_PARTIAL_TASK_ID, "energy_partial");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<double, energy_partial_task>(registrar, "energy_partial");
  }

  return Runtime::start(argc, argv);
}
