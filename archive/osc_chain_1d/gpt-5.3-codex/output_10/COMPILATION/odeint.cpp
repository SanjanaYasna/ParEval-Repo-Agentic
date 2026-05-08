// Translated from HPX to Legion execution model (default mapper).
// Example runs:
//   ./odeint --N 2048 --dt 0.1
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 8   // parallel Legion workers

#include "legion.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

using namespace Legion;

namespace {

constexpr double KAPPA  = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  COMPUTE_DPDT_BLOCK_TASK_ID,
  UPDATE_P_BLOCK_TASK_ID,
  UPDATE_Q_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_Q = 100,
  FID_P,
  FID_DPDT
};

struct ComputeArgs {
  int64_t N;
};

struct UpdateArgs {
  double alpha;
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return std::copysign(std::pow(std::abs(x), k), x);
}

double compute_energy(Runtime* runtime, Context ctx, LogicalRegion lr, IndexSpace is) {
  InlineLauncher launcher(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.add_field(FID_Q);
  launcher.add_field(FID_P);

  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_ONLY, double, 1> q_acc(pr, FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(pr, FID_P);

  Rect<1> rect = runtime->get_index_space_domain(ctx, is);
  const int64_t lo = rect.lo[0];
  const int64_t hi = rect.hi[0];
  const int64_t N  = hi - lo + 1;

  if (N <= 0) {
    runtime->unmap_region(ctx, pr);
    return 0.0;
  }

  double e = 0.0;

  const double q0 = q_acc[Point<1>(lo)];
  e += 0.5 * checked_pow(q0, LAMBDA) / LAMBDA;

  for (int64_t i = lo; i < hi; ++i) {
    const double qi  = q_acc[Point<1>(i)];
    const double qip = q_acc[Point<1>(i + 1)];
    const double pi  = p_acc[Point<1>(i)];

    e += 0.5 * pi * pi
       + checked_pow(qi, KAPPA) / KAPPA
       + checked_pow(qi - qip, LAMBDA) / LAMBDA;
  }

  const double qn = q_acc[Point<1>(hi)];
  const double pn = p_acc[Point<1>(hi)];
  e += 0.5 * pn * pn
     + checked_pow(qn, KAPPA) / KAPPA
     + 0.5 * checked_pow(qn, LAMBDA) / LAMBDA;

  runtime->unmap_region(ctx, pr);
  return e;
}

void compute_dpdt_block_task(const Task* task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime) {
  assert(regions.size() == 2);
  const auto* args = static_cast<const ComputeArgs*>(task->args);
  const int64_t N = args->N;

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> subrect =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(subrect); it(); it++) {
    const int64_t i = (*it)[0];
    const double qi = q_acc[*it];

    const double left =
        (i == 0) ? -signed_pow(qi, LAMBDA - 1.0)
                 : signed_pow(q_acc[Point<1>(i - 1)] - qi, LAMBDA - 1.0);

    const double right =
        (i == N - 1) ? signed_pow(qi, LAMBDA - 1.0)
                     : signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[*it] = -signed_pow(qi, KAPPA - 1.0) + left - right;
  }
}

void update_p_block_task(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime) {
  assert(regions.size() == 2);
  const auto* args = static_cast<const UpdateArgs*>(task->args);

  FieldAccessor<READ_WRITE, double, 1> p_acc(regions[0], FID_P);
  FieldAccessor<READ_ONLY, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> subrect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(subrect); it(); it++) {
    const double updated = p_acc[*it] + args->alpha * dpdt_acc[*it];
    p_acc[*it] = updated;
  }
}

void update_q_block_task(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime) {
  assert(regions.size() == 2);
  const auto* args = static_cast<const UpdateArgs*>(task->args);

  FieldAccessor<READ_WRITE, double, 1> q_acc(regions[0], FID_Q);
  FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_P);

  Rect<1> subrect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());

  for (PointInRectIterator<1> it(subrect); it(); it++) {
    const double updated = q_acc[*it] + args->alpha * p_acc[*it];
    q_acc[*it] = updated;
  }
}

void launch_compute_dpdt(Runtime* runtime, Context ctx, Domain launch_domain,
                         LogicalRegion lr, LogicalPartition lp, int64_t N) {
  ComputeArgs args{N};
  IndexLauncher launcher(COMPUTE_DPDT_BLOCK_TASK_ID, launch_domain,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp, 0 /* identity projection */,
                                                   WRITE_DISCARD, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

void launch_update_p(Runtime* runtime, Context ctx, Domain launch_domain,
                     LogicalRegion lr, LogicalPartition lp, double alpha) {
  UpdateArgs args{alpha};
  IndexLauncher launcher(UPDATE_P_BLOCK_TASK_ID, launch_domain,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_P);

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

void launch_update_q(Runtime* runtime, Context ctx, Domain launch_domain,
                     LogicalRegion lr, LogicalPartition lp, double alpha) {
  UpdateArgs args{alpha};
  IndexLauncher launcher(UPDATE_Q_BLOCK_TASK_ID, launch_domain,
                         TaskArgument(&args, sizeof(args)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements.back().add_field(FID_P);

  runtime->execute_index_space(ctx, launcher);
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx, Runtime* runtime) {
  namespace po = boost::program_options;

  const InputArgs& args = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("help,h", "Show help")
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(desc)
                    .allow_unregistered()
                    .run();
  po::store(parsed, vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::cout << "Legion runtime options can be appended (e.g., -ll:cpu 8).\n";
    return;
  }

  const std::size_t N_in = vm["N"].as<std::size_t>();
  const std::size_t G_in = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (N_in == 0 || G_in == 0) {
    std::cerr << "N and G must be > 0\n";
    return;
  }

  const int64_t N = static_cast<int64_t>(N_in);
  const int64_t G = static_cast<int64_t>(G_in);
  const int64_t M = (N + G - 1) / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << steps
          << ", dt: " << dt << std::endl;

  // Create region [0..N-1] with fields q, p, dpdt
  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, N - 1));

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_P);
    allocator.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Partition into M contiguous blocks (equal partitioning over colors).
  IndexSpace color_is = runtime->create_index_space(ctx, Rect<1>(0, M - 1));
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
  Domain launch_domain = runtime->get_index_space_domain(ctx, color_is);

  // Initialize q=0 and dpdt=0
  const double zero = 0.0;
  runtime->fill_field(ctx, lr, lr, FID_Q, &zero, sizeof(zero));
  runtime->fill_field(ctx, lr, lr, FID_DPDT, &zero, sizeof(zero));

  // Initialize p with deterministic random numbers in [-1,1]
  {
    InlineLauncher init_p(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
    init_p.add_field(FID_P);
    PhysicalRegion pr = runtime->map_region(ctx, init_p);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> p_acc(pr, FID_P);
    Rect<1> rect = runtime->get_index_space_domain(ctx, is);

    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);

    for (PointInRectIterator<1> it(rect); it(); it++) {
      p_acc[*it] = distribution(engine);
    }

    runtime->unmap_region(ctx, pr);
  }

  const long long e_init = static_cast<long long>(std::llround(compute_energy(runtime, ctx, lr, is)));
  outfile << "Initialization complete, energy: " << e_init << std::endl;

  // Symplectic integration (velocity-Verlet):
  // p += 0.5*dt*f(q), q += dt*p, p += 0.5*dt*f(q)
  for (std::size_t s = 0; s < steps; ++s) {
    launch_compute_dpdt(runtime, ctx, launch_domain, lr, lp, N);
    launch_update_p(runtime, ctx, launch_domain, lr, lp, 0.5 * dt);
    launch_update_q(runtime, ctx, launch_domain, lr, lp, dt);
    launch_compute_dpdt(runtime, ctx, launch_domain, lr, lp, N);
    launch_update_p(runtime, ctx, launch_domain, lr, lp, 0.5 * dt);
  }

  const long long e_final = static_cast<long long>(std::llround(compute_energy(runtime, ctx, lr, is)));
  outfile << "Integration complete, energy: " << e_final << std::endl;

  // Cleanup
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
}

} // end anonymous namespace

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(COMPUTE_DPDT_BLOCK_TASK_ID, "compute_dpdt_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<compute_dpdt_block_task>(registrar, "compute_dpdt_block");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_P_BLOCK_TASK_ID, "update_p_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_p_block_task>(registrar, "update_p_block");
  }

  {
    TaskVariantRegistrar registrar(UPDATE_Q_BLOCK_TASK_ID, "update_q_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<update_q_block_task>(registrar, "update_q_block");
  }

  return Runtime::start(argc, argv);
}
