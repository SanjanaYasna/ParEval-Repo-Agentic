// Translated from HPX to Legion execution model.
// Example parallel run:
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 8

#include "legion.h"

#include <boost/program_options.hpp>
#include <boost/numeric/odeint/integrate/integrate_n_steps.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/ref.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;
namespace po = boost::program_options;
using boost::numeric::odeint::integrate_n_steps;
using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan;

using dvec = std::vector<double>;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  COMPUTE_DPDT_TASK_ID
};

enum FieldIDs {
  FID_Q = 100,
  FID_DPDT = 101
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

inline double checked_pow_abs(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signum(double x) {
  return (x > 0.0) - (x < 0.0);
}

inline double signed_pow(double x, double k) {
  return checked_pow_abs(x, k) * signum(x);
}

double energy(const dvec &q, const dvec &p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * checked_pow_abs(q[0], LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i]
       + checked_pow_abs(q[i], KAPPA) / KAPPA
       + checked_pow_abs(q[i] - q[i + 1], LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + checked_pow_abs(q[N - 1], KAPPA) / KAPPA
     + 0.5 * checked_pow_abs(q[N - 1], LAMBDA) / LAMBDA;
  return e;
}

// Legion task: compute dpdt for one block of indices.
void compute_dpdt_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime) {
  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_DPDT);

  Rect<1> full_rect =
      runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  Rect<1> sub_rect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

  const coord_t lo = full_rect.lo[0];
  const coord_t hi = full_rect.hi[0];

  for (coord_t i = sub_rect.lo[0]; i <= sub_rect.hi[0]; ++i) {
    const double qi = q_acc[Point<1>(i)];

    const double onsite = -signed_pow(qi, KAPPA - 1.0);
    const double left_coupling =
        (i == lo) ? -signed_pow(qi, LAMBDA - 1.0)
                  :  signed_pow(q_acc[Point<1>(i - 1)] - qi, LAMBDA - 1.0);

    const double right_coupling =
        (i == hi) ? -signed_pow(qi, LAMBDA - 1.0)
                  : -signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[Point<1>(i)] = onsite + left_coupling + right_coupling;
  }
}

// ODEINT system functor that delegates dpdt computation to Legion index tasks.
struct OscChainLegionSystem {
  Runtime *runtime;
  Context ctx;
  LogicalRegion lr;
  LogicalPartition lp;
  Domain color_domain;
  Rect<1> full_rect;

  OscChainLegionSystem(Runtime *rt, Context c, LogicalRegion reg, LogicalPartition part,
                       Domain cd, Rect<1> fr)
      : runtime(rt), ctx(c), lr(reg), lp(part), color_domain(cd), full_rect(fr) {}

  void operator()(const dvec &q, dvec &dpdt) const {
    // Copy q -> Legion field FID_Q
    {
      RegionRequirement rr(lr, WRITE_DISCARD, EXCLUSIVE, lr);
      rr.add_field(FID_Q);
      InlineLauncher launcher(rr);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      pr.wait_until_valid();

      FieldAccessor<WRITE_DISCARD, double, 1> q_acc(pr, FID_Q);
      for (coord_t i = full_rect.lo[0]; i <= full_rect.hi[0]; ++i) {
        q_acc[Point<1>(i)] = q[static_cast<std::size_t>(i - full_rect.lo[0])];
      }

      runtime->unmap_region(ctx, pr);
    }

    // Parallel dpdt computation by block.
    IndexLauncher il(COMPUTE_DPDT_TASK_ID, color_domain, TaskArgument(nullptr, 0), ArgumentMap());

    il.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
    il.region_requirements[0].add_field(FID_Q);

    il.add_region_requirement(RegionRequirement(lp, 0 /* identity projection */,
                                                WRITE_DISCARD, EXCLUSIVE, lr));
    il.region_requirements[1].add_field(FID_DPDT);

    runtime->execute_index_space(ctx, il);

    // Copy Legion field FID_DPDT -> dpdt
    dpdt.resize(q.size());
    {
      RegionRequirement rr(lr, READ_ONLY, EXCLUSIVE, lr);
      rr.add_field(FID_DPDT);
      InlineLauncher launcher(rr);
      PhysicalRegion pr = runtime->map_region(ctx, launcher);
      pr.wait_until_valid();

      FieldAccessor<READ_ONLY, double, 1> dpdt_acc(pr, FID_DPDT);
      for (coord_t i = full_rect.lo[0]; i <= full_rect.hi[0]; ++i) {
        dpdt[static_cast<std::size_t>(i - full_rect.lo[0])] = dpdt_acc[Point<1>(i)];
      }

      runtime->unmap_region(ctx, pr);
    }
  }

  // Optional overload if a time argument is requested.
  void operator()(const dvec &q, dvec &dpdt, const double /*t*/) const {
    (*this)(q, dpdt);
  }
};

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
  const InputArgs &input_args = Runtime::get_input_args();

  po::options_description desc_commandline("Usage: odeint [options]");
  desc_commandline.add_options()
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  try {
    auto parsed = po::command_line_parser(input_args.argc, input_args.argv)
                      .options(desc_commandline)
                      .allow_unregistered() // keep Legion runtime args intact
                      .run();
    po::store(parsed, vm);
    po::notify(vm);
  } catch (const std::exception &e) {
    std::cerr << "Error parsing command line: " << e.what() << std::endl;
    return;
  }

  const std::size_t N = vm["N"].as<std::size_t>();
  const std::size_t G = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (N == 0 || G == 0) {
    std::cerr << "N and G must be > 0." << std::endl;
    return;
  }
  if (N % G != 0) {
    std::cerr << "This implementation requires N % G == 0 (got N=" << N
              << ", G=" << G << ")." << std::endl;
    return;
  }

  const std::size_t M = N / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing." << std::endl;
    return;
  }

  outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt
          << std::endl;

  // Host-side state vectors (used by ODEINT stepper)
  dvec q(N, 0.0), p(N, 0.0);

  std::mt19937 engine(0);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (std::size_t i = 0; i < N; ++i) p[i] = distribution(engine);

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p))) << std::endl;

  // Legion region resources used by system functor.
  const Rect<1> elem_rect(Point<1>(0), Point<1>(static_cast<coord_t>(N - 1)));
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_Q);
    allocator.allocate_field(sizeof(double), FID_DPDT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  // Partition into contiguous blocks of size G.
  IndexPartition ip = runtime->create_partition_by_blockify(
      ctx, is, Point<1>(static_cast<coord_t>(G)));
  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
  Domain color_domain = runtime->get_index_partition_color_space(ctx, ip);

  // ODE integration (same stepper family as original HPX code).
  using stepper_type =
      symplectic_rkn_sb3a_mclachlan<dvec, dvec, double, dvec, dvec, double>;

  OscChainLegionSystem system(runtime, ctx, lr, lp, color_domain, elem_rect);

  integrate_n_steps(stepper_type(), system,
                    std::make_pair(boost::ref(q), boost::ref(p)),
                    0.0, dt, steps);

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p))) << std::endl;

  // Cleanup Legion resources.
  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_index_partition(ctx, ip);
  runtime->destroy_field_space(ctx, fs);
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

  return Runtime::start(argc, argv);
}
