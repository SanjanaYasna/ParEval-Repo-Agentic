// Translated from HPX to Legion execution model.
// No custom mapper required (default mapper assumed).

#include "legion.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace Legion;
namespace po = boost::program_options;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1000,
  COMPUTE_FORCE_TASK_ID,
  KICK_TASK_ID,
  DRIFT_TASK_ID
};

enum FieldIDs {
  FID_Q = 1,
  FID_P,
  FID_DPDT
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct ScaleArgs {
  double factor;
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  return checked_pow(x, k) * s;
}

static double compute_energy(Context ctx, Runtime *runtime, LogicalRegion lr, IndexSpace is) {
  RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
  req.add_field(FID_Q);
  req.add_field(FID_P);

  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();

  const FieldAccessor<READ_ONLY, double, 1> q_acc(pr, FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p_acc(pr, FID_P);

  Rect<1> rect = runtime->get_index_space_domain(ctx, is);
  const coord_t lo = rect.lo[0];
  const coord_t hi = rect.hi[0];

  double e = 0.0;
  if (lo <= hi) {
    const double q0 = q_acc[Point<1>(lo)];
    e = 0.5 * checked_pow(q0, LAMBDA) / LAMBDA;

    for (coord_t i = lo; i < hi; ++i) {
      const double qi = q_acc[Point<1>(i)];
      const double qip1 = q_acc[Point<1>(i + 1)];
      const double pi = p_acc[Point<1>(i)];

      e += 0.5 * pi * pi
         + checked_pow(qi, KAPPA) / KAPPA
         + checked_pow(qi - qip1, LAMBDA) / LAMBDA;
    }

    const double qn = q_acc[Point<1>(hi)];
    const double pn = p_acc[Point<1>(hi)];
    e += 0.5 * pn * pn
       + checked_pow(qn, KAPPA) / KAPPA
       + 0.5 * checked_pow(qn, LAMBDA) / LAMBDA;
  }

  runtime->unmap_region(ctx, pr);
  return e;
}

void compute_force_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime) {
  // regions[0]: full q (READ_ONLY)
  // regions[1]: local dpdt block (WRITE_DISCARD)
  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> full_rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> my_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  const coord_t glo = full_rect.lo[0];
  const coord_t ghi = full_rect.hi[0];

  for (coord_t i = my_rect.lo[0]; i <= my_rect.hi[0]; ++i) {
    const double qi = q_acc[Point<1>(i)];

    const double left_term = (i == glo)
      ? -signed_pow(qi, LAMBDA - 1.0)
      :  signed_pow(q_acc[Point<1>(i - 1)] - qi, LAMBDA - 1.0);

    const double right_term = (i == ghi)
      ?  signed_pow(qi, LAMBDA - 1.0)
      :  signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[Point<1>(i)] = -signed_pow(qi, KAPPA - 1.0) + left_term - right_term;
  }
}

void kick_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime) {
  // p += factor * dpdt on local block
  const auto *args = static_cast<const ScaleArgs *>(task->args);
  const double factor = args->factor;

  FieldAccessor<READ_WRITE, double, 1> p_acc(regions[0], FID_P);
  const FieldAccessor<READ_ONLY, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> pt(i);
    const double pv = p_acc[pt];
    p_acc[pt] = pv + factor * dpdt_acc[pt];
  }
}

void drift_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  // q += factor * p on local block
  const auto *args = static_cast<const ScaleArgs *>(task->args);
  const double factor = args->factor;

  FieldAccessor<READ_WRITE, double, 1> q_acc(regions[0], FID_Q);
  const FieldAccessor<READ_ONLY, double, 1> p_acc(regions[1], FID_P);

  Rect<1> rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());

  for (coord_t i = rect.lo[0]; i <= rect.hi[0]; ++i) {
    const Point<1> pt(i);
    const double qv = q_acc[pt];
    q_acc[pt] = qv + factor * p_acc[pt];
  }
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
  // Parse CLI options (ignore unknown Legion runtime flags, e.g., -ll:cpu)
  const InputArgs &input_args = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  auto parsed = po::command_line_parser(input_args.argc, input_args.argv)
                    .options(desc)
                    .allow_unregistered()
                    .run();
  po::store(parsed, vm);
  po::notify(vm);

  const std::size_t N = vm["N"].as<std::size_t>();
  const std::size_t G = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (N == 0 || G == 0 || (N % G) != 0) {
    std::cerr << "Error: require N > 0, G > 0, and N % G == 0.\n";
    return;
  }

  const std::size_t M = N / G;

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

  // Create main data region: q, p, dpdt
  Rect<1> elem_rect(Point<1>(0), Point<1>(static_cast<coord_t>(N - 1)));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_P);
    allocator.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Partition into M equal contiguous blocks (size G)
  Rect<1> color_rect(Point<1>(0), Point<1>(static_cast<coord_t>(M - 1)));
  IndexSpace color_is = runtime->create_index_space(ctx, color_rect);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  // Initialize q=0, p=random[-1,1], dpdt=0
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
    FieldAccessor<WRITE_DISCARD, double, 1> d_acc(pr, FID_DPDT);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::mt19937 gen(0);

    for (coord_t i = elem_rect.lo[0]; i <= elem_rect.hi[0]; ++i) {
      Point<1> p(i);
      q_acc[p] = 0.0;
      p_acc[p] = dist(gen);
      d_acc[p] = 0.0;
    }

    runtime->unmap_region(ctx, pr);
  }

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(compute_energy(ctx, runtime, lr, is)))
          << std::endl;

  const Domain launch_domain(color_rect);

  auto launch_compute_force = [&]() {
    IndexLauncher launcher(COMPUTE_FORCE_TASK_ID, launch_domain, TaskArgument(nullptr, 0), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_Q);

    launcher.add_region_requirement(RegionRequirement(lp, 0 /*proj*/, WRITE_DISCARD, EXCLUSIVE, lr));
    launcher.region_requirements[1].add_field(FID_DPDT);

    runtime->execute_index_space(ctx, launcher).wait_all_results();
  };

  auto launch_kick = [&](double factor) {
    ScaleArgs args{factor};
    IndexLauncher launcher(KICK_TASK_ID, launch_domain, TaskArgument(&args, sizeof(args)), ArgumentMap());

    launcher.add_region_requirement(RegionRequirement(lp, 0 /*proj*/, READ_WRITE, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_P);

    launcher.add_region_requirement(RegionRequirement(lp, 0 /*proj*/, READ_ONLY, EXCLUSIVE, lr));
    launcher.region_requirements[1].add_field(FID_DPDT);

    runtime->execute_index_space(ctx, launcher).wait_all_results();
  };

  auto launch_drift = [&](double factor) {
    ScaleArgs args{factor};
    IndexLauncher launcher(DRIFT_TASK_ID, launch_domain, TaskArgument(&args, sizeof(args)), ArgumentMap());

    launcher.add_region_requirement(RegionRequirement(lp, 0 /*proj*/, READ_WRITE, EXCLUSIVE, lr));
    launcher.region_requirements[0].add_field(FID_Q);

    launcher.add_region_requirement(RegionRequirement(lp, 0 /*proj*/, READ_ONLY, EXCLUSIVE, lr));
    launcher.region_requirements[1].add_field(FID_P);

    runtime->execute_index_space(ctx, launcher).wait_all_results();
  };

  // Time integration (Legion parallel over blocks): velocity-Verlet
  for (std::size_t s = 0; s < steps; ++s) {
    launch_compute_force();
    launch_kick(0.5 * dt);
    launch_drift(dt);
    launch_compute_force();
    launch_kick(0.5 * dt);
  }

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(compute_energy(ctx, runtime, lr, is)))
          << std::endl;

  // Cleanup
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_partition(ctx, ip);
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
    TaskVariantRegistrar registrar(KICK_TASK_ID, "kick");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<kick_task>(registrar, "kick");
  }
  {
    TaskVariantRegistrar registrar(DRIFT_TASK_ID, "drift");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<drift_task>(registrar, "drift");
  }

  return Runtime::start(argc, argv);
}
