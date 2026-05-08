// Translated to Legion execution model (default mapper)
// Original HPX version integrated a nonlinear 1D oscillator chain.

#include "legion.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

using namespace Legion;

namespace po = boost::program_options;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  INIT_BLOCK_TASK_ID,
  COMPUTE_DPDT_TASK_ID,
  UPDATE_P_TASK_ID,
  UPDATE_Q_TASK_ID
};

enum FieldIDs {
  FID_Q = 100,
  FID_P,
  FID_DPDT
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct InitArgs {
  std::uint64_t seed;
};

struct ComputeArgs {
  std::int64_t N;
};

struct UpdateArgs {
  double factor;
};

inline double abs_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double y) {
  return abs_pow(x, y) * ((x > 0.0) - (x < 0.0));
}

void init_block_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime) {
  const InitArgs &args = *reinterpret_cast<const InitArgs *>(task->args);

  FieldAccessor<WRITE_DISCARD, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<WRITE_DISCARD, double, 1> p_acc(regions[0], FID_P);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[0], FID_DPDT);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  std::mt19937 eng(static_cast<std::uint32_t>(args.seed));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> p(i);
    q_acc[p] = 0.0;
    p_acc[p] = dist(eng);
    dpdt_acc[p] = 0.0;
  }
}

void compute_dpdt_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
  const ComputeArgs &args = *reinterpret_cast<const ComputeArgs *>(task->args);

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> pi(i);
    const double qi = q_acc[pi];

    const double coupling_left =
        (i == 0) ? 0.5 * signed_pow(qi, LAMBDA - 1.0)
                 : signed_pow(qi - q_acc[Point<1>(i - 1)], LAMBDA - 1.0);

    const double coupling_right =
        (i == args.N - 1) ? 0.5 * signed_pow(qi, LAMBDA - 1.0)
                          : signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[pi] = -signed_pow(qi, KAPPA - 1.0) - coupling_left - coupling_right;
  }
}

void update_p_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  (void)ctx;
  (void)runtime;

  const UpdateArgs &args = *reinterpret_cast<const UpdateArgs *>(task->args);

  FieldAccessor<READ_WRITE, double, 1> p_acc(regions[0], FID_P);
  FieldAccessor<READ_ONLY, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> p(i);
    const double cur = p_acc[p];
    p_acc[p] = cur + args.factor * dpdt_acc[p];
  }
}

void update_q_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime) {
  (void)ctx;
  (void)runtime;

  const UpdateArgs &args = *reinterpret_cast<const UpdateArgs *>(task->args);

  FieldAccessor<READ_WRITE, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_P);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> p(i);
    const double cur = q_acc[p];
    q_acc[p] = cur + args.factor * p_acc[p];
  }
}

double compute_energy(Context ctx, Runtime *runtime, LogicalRegion lr, std::size_t N) {
  InlineLauncher launcher(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.requirement.add_field(FID_Q);
  launcher.requirement.add_field(FID_P);

  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_ONLY, double, 1> q_acc(pr, FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(pr, FID_P);

  auto q = [&](std::size_t i) { return q_acc[Point<1>(static_cast<coord_t>(i))]; };
  auto p = [&](std::size_t i) { return p_acc[Point<1>(static_cast<coord_t>(i))]; };

  double e = 0.0;
  e += 0.5 * abs_pow(q(0), LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i + 1 < N; ++i) {
    e += 0.5 * p(i) * p(i);
    e += abs_pow(q(i), KAPPA) / KAPPA;
    e += abs_pow(q(i) - q(i + 1), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p(N - 1) * p(N - 1);
  e += abs_pow(q(N - 1), KAPPA) / KAPPA;
  e += 0.5 * abs_pow(q(N - 1), LAMBDA) / LAMBDA;

  runtime->unmap_region(ctx, pr);
  return e;
}

static void launch_compute_dpdt(Context ctx, Runtime *runtime, Domain colors,
                                LogicalRegion lr, LogicalPartition lp, std::int64_t N) {
  ComputeArgs args{N};
  IndexLauncher launcher(COMPUTE_DPDT_TASK_ID, colors,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp, 0 /* projection */, WRITE_DISCARD,
                                                   EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

static void launch_update_p(Context ctx, Runtime *runtime, Domain colors,
                            LogicalRegion lr, LogicalPartition lp, double factor) {
  UpdateArgs args{factor};
  IndexLauncher launcher(UPDATE_P_TASK_ID, colors,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(
      RegionRequirement(lp, 0 /* projection */, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_P);

  launcher.add_region_requirement(
      RegionRequirement(lp, 0 /* projection */, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

static void launch_update_q(Context ctx, Runtime *runtime, Domain colors,
                            LogicalRegion lr, LogicalPartition lp, double factor) {
  UpdateArgs args{factor};
  IndexLauncher launcher(UPDATE_Q_TASK_ID, colors,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(
      RegionRequirement(lp, 0 /* projection */, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_Q);

  launcher.add_region_requirement(
      RegionRequirement(lp, 0 /* projection */, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_P);

  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime) {
  (void)task;
  (void)regions;

  // CLI compatible with original usage:
  // ./odeint --N 2048 --dt 0.1
  const InputArgs &in_args = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("help", "print help")
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  try {
    auto parsed =
        po::command_line_parser(in_args.argc, in_args.argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    po::notify(vm);
  } catch (const std::exception &e) {
    std::cerr << "Failed to parse command line: " << e.what() << "\n";
    return;
  }

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return;
  }

  const std::size_t N = vm["N"].as<std::size_t>();
  const std::size_t G = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (N == 0 || G == 0) {
    std::cerr << "N and G must be > 0\n";
    return;
  }

  const std::size_t M = (N + G - 1) / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt
          << std::endl;

  // Build Legion data model
  IndexSpace is = runtime->create_index_space(
      ctx, Rect<1>(0, static_cast<coord_t>(N - 1)));

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
    alloc.allocate_field(sizeof(double), FID_Q);
    alloc.allocate_field(sizeof(double), FID_P);
    alloc.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  IndexPartition ip =
      runtime->create_partition_by_blockify(ctx, is, Point<1>(static_cast<coord_t>(G)));
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  Domain color_domain(Rect<1>(0, static_cast<coord_t>(M - 1)));

  // Initialization (q=0, p=random, dpdt=0) using a single deterministic task.
  InitArgs init_args{5489u};
  TaskLauncher init_launcher(INIT_BLOCK_TASK_ID,
                             TaskArgument(&init_args, sizeof(init_args)));
  init_launcher.add_region_requirement(
      RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
  init_launcher.region_requirements.back().add_field(FID_Q);
  init_launcher.region_requirements.back().add_field(FID_P);
  init_launcher.region_requirements.back().add_field(FID_DPDT);

  runtime->execute_task(ctx, init_launcher).get_void_result();

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(compute_energy(ctx, runtime, lr, N)))
          << std::endl;

  // Velocity-Verlet integration in Legion tasks
  for (std::size_t s = 0; s < steps; ++s) {
    launch_compute_dpdt(ctx, runtime, color_domain, lr, lp, static_cast<std::int64_t>(N));
    launch_update_p(ctx, runtime, color_domain, lr, lp, 0.5 * dt);
    launch_update_q(ctx, runtime, color_domain, lr, lp, dt);
    launch_compute_dpdt(ctx, runtime, color_domain, lr, lp, static_cast<std::int64_t>(N));
    launch_update_p(ctx, runtime, color_domain, lr, lp, 0.5 * dt);
  }

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(compute_energy(ctx, runtime, lr, N)))
          << std::endl;

  // Cleanup
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar reg(TOP_LEVEL_TASK_ID, "top_level");
    reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(reg, "top_level");
  }
  {
    TaskVariantRegistrar reg(INIT_BLOCK_TASK_ID, "init_block");
    reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    reg.set_leaf();
    Runtime::preregister_task_variant<init_block_task>(reg, "init_block");
  }
  {
    TaskVariantRegistrar reg(COMPUTE_DPDT_TASK_ID, "compute_dpdt");
    reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    reg.set_leaf();
    Runtime::preregister_task_variant<compute_dpdt_task>(reg, "compute_dpdt");
  }
  {
    TaskVariantRegistrar reg(UPDATE_P_TASK_ID, "update_p");
    reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    reg.set_leaf();
    Runtime::preregister_task_variant<update_p_task>(reg, "update_p");
  }
  {
    TaskVariantRegistrar reg(UPDATE_Q_TASK_ID, "update_q");
    reg.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    reg.set_leaf();
    Runtime::preregister_task_variant<update_q_task>(reg, "update_q");
  }

  return Runtime::start(argc, argv);
}
