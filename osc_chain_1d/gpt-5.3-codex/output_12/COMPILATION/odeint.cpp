// Translated from HPX to Legion execution model (default mapper)
// odeint.cpp

#include "legion.h"

#include <boost/program_options.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace Legion;
namespace po = boost::program_options;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  FORCE_BLOCK_TASK_ID,
  UPDATE_P_BLOCK_TASK_ID,
  UPDATE_Q_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_Q = 1,
  FID_P,
  FID_DPDT
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

inline double pow_abs(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return std::copysign(std::pow(std::abs(x), k), x);
}

static void force_block_task(const Task* task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime) {
  // regions[0]: full region, READ_ONLY, FID_Q
  // regions[1]: block subregion, WRITE_DISCARD, FID_DPDT
  const FieldAccessor<READ_ONLY, double, 1> q(regions[0], FID_Q);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt(regions[1], FID_DPDT);

  const Rect<1> q_rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  const Rect<1> my_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  const coord_t lo = q_rect.lo[0];
  const coord_t hi = q_rect.hi[0];

  for (coord_t i = my_rect.lo[0]; i <= my_rect.hi[0]; ++i) {
    const double qi = q[Point<1>(i)];
    const double ql = (i == lo) ? 0.0 : q[Point<1>(i - 1)];
    const double qr = (i == hi) ? 0.0 : q[Point<1>(i + 1)];

    // dpdt[i] = -|qi|^(KAPPA-1)sign(qi)
    //           + |ql-qi|^(LAMBDA-1)sign(ql-qi)
    //           - |qi-qr|^(LAMBDA-1)sign(qi-qr)
    dpdt[Point<1>(i)] =
        -signed_pow(qi, KAPPA - 1.0) +
        signed_pow(ql - qi, LAMBDA - 1.0) -
        signed_pow(qi - qr, LAMBDA - 1.0);
  }
}

static void update_p_block_task(const Task* task,
                                const std::vector<PhysicalRegion>& regions,
                                Context ctx, Runtime* runtime) {
  // regions[0]: p block, READ_WRITE
  // regions[1]: dpdt block, READ_ONLY
  const double alpha = *reinterpret_cast<const double*>(task->args);

  const FieldAccessor<READ_WRITE, double, 1> p(regions[0], FID_P);
  const FieldAccessor<READ_ONLY, double, 1> dpdt(regions[1], FID_DPDT);

  const Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> pt(i);
    const double pv = p[pt];
    p[pt] = pv + alpha * dpdt[pt];
  }
}

static void update_q_block_task(const Task* task,
                                const std::vector<PhysicalRegion>& regions,
                                Context ctx, Runtime* runtime) {
  // regions[0]: q block, READ_WRITE
  // regions[1]: p block, READ_ONLY
  const double alpha = *reinterpret_cast<const double*>(task->args);

  const FieldAccessor<READ_WRITE, double, 1> q(regions[0], FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p(regions[1], FID_P);

  const Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> pt(i);
    const double qv = q[pt];
    q[pt] = qv + alpha * p[pt];
  }
}

static void launch_force_blocks(Runtime* runtime, Context ctx,
                                LogicalRegion lr, LogicalPartition lp,
                                const Domain& color_domain) {
  IndexLauncher launcher(FORCE_BLOCK_TASK_ID, color_domain, TaskArgument(nullptr, 0), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

static void launch_update_p_blocks(Runtime* runtime, Context ctx,
                                   LogicalRegion lr, LogicalPartition lp,
                                   const Domain& color_domain, double alpha) {
  IndexLauncher launcher(UPDATE_P_BLOCK_TASK_ID, color_domain, TaskArgument(&alpha, sizeof(alpha)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_P);

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_DPDT);

  runtime->execute_index_space(ctx, launcher);
}

static void launch_update_q_blocks(Runtime* runtime, Context ctx,
                                   LogicalRegion lr, LogicalPartition lp,
                                   const Domain& color_domain, double alpha) {
  IndexLauncher launcher(UPDATE_Q_BLOCK_TASK_ID, color_domain, TaskArgument(&alpha, sizeof(alpha)), ArgumentMap());

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_WRITE, EXCLUSIVE, lr));
  launcher.region_requirements[0].add_field(FID_Q);

  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, lr));
  launcher.region_requirements[1].add_field(FID_P);

  runtime->execute_index_space(ctx, launcher);
}

static double compute_energy(Runtime* runtime, Context ctx, LogicalRegion lr, std::size_t N) {
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(FID_Q);
  req.add_field(FID_P);

  InlineLauncher inline_launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, inline_launcher);
  pr.wait_until_valid();

  const FieldAccessor<READ_ONLY, double, 1> q(pr, FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p(pr, FID_P);

  double e = 0.0;
  if (N > 0) {
    e += 0.5 * pow_abs(q[Point<1>(0)], LAMBDA) / LAMBDA;

    for (std::size_t i = 0; i + 1 < N; ++i) {
      const double qi = q[Point<1>(static_cast<coord_t>(i))];
      const double qip1 = q[Point<1>(static_cast<coord_t>(i + 1))];
      const double pi = p[Point<1>(static_cast<coord_t>(i))];
      e += 0.5 * pi * pi
         + pow_abs(qi, KAPPA) / KAPPA
         + pow_abs(qi - qip1, LAMBDA) / LAMBDA;
    }

    const std::size_t last = N - 1;
    const double ql = q[Point<1>(static_cast<coord_t>(last))];
    const double pl = p[Point<1>(static_cast<coord_t>(last))];
    e += 0.5 * pl * pl
       + pow_abs(ql, KAPPA) / KAPPA
       + 0.5 * pow_abs(ql, LAMBDA) / LAMBDA;
  }

  runtime->unmap_region(ctx, pr);
  return e;
}

static void top_level_task(const Task*,
                           const std::vector<PhysicalRegion>&,
                           Context ctx, Runtime* runtime) {
  const InputArgs& in = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)")
      ("help", "Print help");

  po::variables_map vm;
  try {
    auto parsed = po::command_line_parser(in.argc, in.argv)
                      .options(desc)
                      .allow_unregistered() // ignore Legion runtime args, e.g., -ll:cpu
                      .run();
    po::store(parsed, vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Command line parse error: " << e.what() << "\n";
    return;
  }

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return;
  }

  const std::size_t N_in = vm["N"].as<std::size_t>();
  const std::size_t G = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (G == 0) {
    std::cerr << "Error: G must be > 0\n";
    return;
  }

  const std::size_t M = N_in / G;
  const std::size_t N = M * G; // matches original block-based behavior

  if (M == 0 || N == 0) {
    std::cerr << "Error: N/G must be >= 1\n";
    return;
  }

  if (N != N_in) {
    std::cerr << "Warning: N is not divisible by G. Using N' = " << N
              << " (truncating tail elements).\n";
  }

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << N_in
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << steps
          << ", dt: " << dt << "\n";

  // Legion data model
  const Rect<1> elem_rect(0, static_cast<coord_t>(N - 1));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_P);
    allocator.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  const Rect<1> color_rect(0, static_cast<coord_t>(M - 1));
  IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
  const Domain color_domain = runtime->get_index_space_domain(ctx, color_is);

  // Initialize q=0 and dpdt=0
  const double zero = 0.0;
  runtime->fill_field(ctx, lr, lr, FID_Q, zero);
  runtime->fill_field(ctx, lr, lr, FID_DPDT, zero);

  // Initialize p with deterministic random values
  std::vector<double> p_init(N);
  std::mt19937 eng(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < N; ++i) p_init[i] = dist(eng);

  {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_P);
    InlineLauncher init_p(req);
    PhysicalRegion pr = runtime->map_region(ctx, init_p);
    pr.wait_until_valid();

    const FieldAccessor<WRITE_DISCARD, double, 1> p(pr, FID_P);
    for (std::size_t i = 0; i < N; ++i)
      p[Point<1>(static_cast<coord_t>(i))] = p_init[i];

    runtime->unmap_region(ctx, pr);
  }

  // Initial energy
  const double e0 = compute_energy(runtime, ctx, lr, N);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(e0)) << "\n";

  // Symplectic integration (velocity-Verlet), block-parallel via Legion index tasks
  for (std::size_t s = 0; s < steps; ++s) {
    launch_force_blocks(runtime, ctx, lr, lp, color_domain);
    launch_update_p_blocks(runtime, ctx, lr, lp, color_domain, 0.5 * dt);
    launch_update_q_blocks(runtime, ctx, lr, lp, color_domain, dt);
    launch_force_blocks(runtime, ctx, lr, lp, color_domain);
    launch_update_p_blocks(runtime, ctx, lr, lp, color_domain, 0.5 * dt);
  }

  const double e1 = compute_energy(runtime, ctx, lr, N);
  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(e1)) << "\n";

  // Cleanup
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip);
  runtime->destroy_index_space(ctx, color_is);
  runtime->destroy_index_space(ctx, is);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(FORCE_BLOCK_TASK_ID, "force_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<force_block_task>(registrar, "force_block");
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
