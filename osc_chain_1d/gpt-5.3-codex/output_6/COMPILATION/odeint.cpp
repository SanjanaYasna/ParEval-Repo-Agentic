// Translated from HPX to Legion execution model
// Example Legion run (4 CPU workers):
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 4

#include "legion.h"

#include <boost/program_options.hpp>
#include <boost/ref.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;
using state_type = std::vector<double>;

namespace {

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  FORCE_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_Q = 100,
  FID_DPDT
};

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

// Globals used by rhs function called by boost::odeint
Runtime *g_runtime = nullptr;
Context g_ctx = nullptr;
LogicalRegion g_lr;
LogicalPartition g_lp;
std::size_t g_N = 0;
std::size_t g_M = 0;

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

double energy(const state_type &q, const state_type &p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * checked_pow(std::abs(q[0]), LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i + 1 < N; ++i) {
    e += 0.5 * p[i] * p[i]
       + checked_pow(q[i], KAPPA) / KAPPA
       + checked_pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + checked_pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * checked_pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
  return e;
}

void write_q_field(const state_type &q) {
  RegionRequirement req(g_lr, WRITE_DISCARD, EXCLUSIVE, g_lr);
  req.add_field(FID_Q);
  InlineLauncher launcher(req);
  PhysicalRegion pr = g_runtime->map_region(g_ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_Q);
  for (std::size_t i = 0; i < q.size(); ++i) {
    acc[Point<1>(static_cast<coord_t>(i))] = q[i];
  }

  g_runtime->unmap_region(g_ctx, pr);
}

void read_dpdt_field(state_type &dpdt) {
  RegionRequirement req(g_lr, READ_ONLY, EXCLUSIVE, g_lr);
  req.add_field(FID_DPDT);
  InlineLauncher launcher(req);
  PhysicalRegion pr = g_runtime->map_region(g_ctx, launcher);
  pr.wait_until_valid();

  FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_DPDT);
  dpdt.resize(g_N);
  for (std::size_t i = 0; i < g_N; ++i) {
    dpdt[i] = acc[Point<1>(static_cast<coord_t>(i))];
  }

  g_runtime->unmap_region(g_ctx, pr);
}

// RHS called by boost::odeint stepper; internally launches Legion parallel tasks.
void osc_chain(const state_type &q, state_type &dpdt) {
  write_q_field(q);

  ArgumentMap arg_map;
  IndexLauncher launcher(
      FORCE_BLOCK_TASK_ID,
      Domain(Rect<1>(0, static_cast<coord_t>(g_M - 1))),
      TaskArgument(nullptr, 0),
      arg_map);

  // Full q (read-only) visible to each point task
  launcher.add_region_requirement(RegionRequirement(g_lr, READ_ONLY, EXCLUSIVE, g_lr));
  launcher.region_requirements.back().add_field(FID_Q);

  // Point task writes only its block in dpdt
  launcher.add_region_requirement(
      RegionRequirement(g_lp, 0 /* identity projection */, WRITE_DISCARD, EXCLUSIVE, g_lr));
  launcher.region_requirements.back().add_field(FID_DPDT);

  FutureMap fm = g_runtime->execute_index_space(g_ctx, launcher);
  fm.wait_all_results();

  read_dpdt_field(dpdt);
}

void force_block_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime *runtime) {
  const FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_Q);
  const FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_DPDT);

  const Rect<1> full_rect =
      runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  const Rect<1> blk_rect =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

  const coord_t lo = full_rect.lo[0];
  const coord_t hi = full_rect.hi[0];

  for (coord_t i = blk_rect.lo[0]; i <= blk_rect.hi[0]; ++i) {
    const double qi = q_acc[Point<1>(i)];

    const double left_term = (i == lo)
      ? -signed_pow(qi, LAMBDA - 1.0)
      :  signed_pow(q_acc[Point<1>(i - 1)] - qi, LAMBDA - 1.0);

    const double right_term = (i == hi)
      ?  signed_pow(qi, LAMBDA - 1.0)
      :  signed_pow(qi - q_acc[Point<1>(i + 1)], LAMBDA - 1.0);

    dpdt_acc[Point<1>(i)] =
      -signed_pow(qi, KAPPA - 1.0) + left_term - right_term;
  }
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx, Runtime *runtime) {
  g_runtime = runtime;
  g_ctx = ctx;

  // Parse app args while allowing Legion runtime args.
  namespace po = boost::program_options;
  const InputArgs &args = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)")
      ("help", "print help");

  po::variables_map vm;
  auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(desc)
                    .allow_unregistered()
                    .run();
  po::store(parsed, vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return;
  }

  const std::size_t N = vm["N"].as<std::size_t>();
  const std::size_t G = vm["G"].as<std::size_t>();
  const std::size_t steps = vm["steps"].as<std::size_t>();
  const double dt = vm["dt"].as<double>();

  if (N == 0 || G == 0) {
    std::cerr << "N and G must be > 0" << std::endl;
    return;
  }

  g_N = N;
  g_M = (N + G - 1) / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing." << std::endl;
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << g_M
          << ", steps: " << steps
          << ", dt: " << dt << std::endl;

  // Create region for q and dpdt
  IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, static_cast<coord_t>(N - 1)));
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator alloc = runtime->create_field_allocator(ctx, fs);
    alloc.allocate_field(sizeof(double), FID_Q);
    alloc.allocate_field(sizeof(double), FID_DPDT);
  }
  g_lr = runtime->create_logical_region(ctx, is, fs);

  Rect<1> color_bounds(0, static_cast<coord_t>(g_M - 1));
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  Transform<1, 1> transform;
  transform[0][0] = static_cast<coord_t>(G);
  Rect<1> extent(0, static_cast<coord_t>(G - 1));

  IndexPartition ip =
      runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);
  g_lp = runtime->get_logical_partition(ctx, g_lr, ip);

  // Initialize q and p
  state_type q_in(N, 0.0), p_in(N, 0.0);
  std::mt19937 engine(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < N; ++i) p_in[i] = dist(engine);

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(energy(q_in, p_in)))
          << std::endl;

  using stepper_type = boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan<state_type>;
  stepper_type stepper;
  auto state = std::make_pair(boost::ref(q_in), boost::ref(p_in));
  double t = 0.0;
  for (std::size_t step = 0; step < steps; ++step) {
    stepper.do_step(osc_chain, state, t, dt);
    t += dt;
  }

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(energy(q_in, p_in)))
          << std::endl;

  // Cleanup Legion resources
  runtime->destroy_logical_region(ctx, g_lr);
  runtime->destroy_index_partition(ctx, ip);
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
    TaskVariantRegistrar registrar(FORCE_BLOCK_TASK_ID, "force_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<force_block_task>(registrar, "force_block");
  }

  return Runtime::start(argc, argv);
}
