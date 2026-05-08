// Translated from HPX to Legion execution model (default mapper).
// Example run with Legion parallelism:
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 4

#include "legion.h"

#include <boost/ref.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;
namespace odeint = boost::numeric::odeint;
namespace po = boost::program_options;

using dvec = std::vector<double>;

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  OSC_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_VAL = 100
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

double energy(const dvec &q, const dvec &p) {
  const size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * checked_pow(std::abs(q[0]), LAMBDA) / LAMBDA;
  for (size_t i = 0; i < N - 1; ++i) {
    e += 0.5 * p[i] * p[i]
       + checked_pow(q[i], KAPPA) / KAPPA
       + checked_pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + checked_pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * checked_pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
  return e;
}

struct OscTaskArgs {
  int64_t N;
};

void osc_block_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime *runtime) {
  assert(task->args != nullptr);
  assert(task->arglen == sizeof(OscTaskArgs));
  assert(regions.size() == 2);

  const auto *args = reinterpret_cast<const OscTaskArgs *>(task->args);

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

  Rect<1> subrect =
      runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space())
          .bounds<1, coord_t>();

  for (PointInRectIterator<1> pir(subrect); pir(); ++pir) {
    const coord_t i = (*pir)[0];
    const double qi = q_acc[*pir];
    const double ql = (i > 0) ? q_acc[Point<1>(i - 1)] : 0.0;
    const double qr = (i + 1 < args->N) ? q_acc[Point<1>(i + 1)] : 0.0;

    dpdt_acc[*pir] =
        -signed_pow(qi, KAPPA - 1.0)
        + signed_pow(ql - qi, LAMBDA - 1.0)
        - signed_pow(qi - qr, LAMBDA - 1.0);
  }
}

class LegionOscChain {
public:
  LegionOscChain(Runtime *rt, Context c, size_t n, size_t g)
      : runtime(rt), ctx(c), N(n), G(g), M(n / g) {
    Rect<1> elem_rect(0, static_cast<coord_t>(N - 1));
    is = runtime->create_index_space(ctx, elem_rect);

    fs = runtime->create_field_space(ctx);
    {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(double), FID_VAL);
    }

    lr_q = runtime->create_logical_region(ctx, is, fs);
    lr_dpdt = runtime->create_logical_region(ctx, is, fs);

    Rect<1> color_rect(0, static_cast<coord_t>(M - 1));
    color_is = runtime->create_index_space(ctx, color_rect);
    ip_dpdt = runtime->create_equal_partition(ctx, is, color_is);
    lp_dpdt = runtime->get_logical_partition(ctx, lr_dpdt, ip_dpdt);
  }

  LegionOscChain(const LegionOscChain &) = delete;
  LegionOscChain &operator=(const LegionOscChain &) = delete;

  ~LegionOscChain() {
    runtime->destroy_logical_region(ctx, lr_q);
    runtime->destroy_logical_region(ctx, lr_dpdt);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_partition(ctx, ip_dpdt);
    runtime->destroy_index_space(ctx, color_is);
    runtime->destroy_index_space(ctx, is);
  }

  void operator()(const dvec &q, dvec &dpdt) const {
    operator()(q, dpdt, 0.0);
  }

  void operator()(const dvec &q, dvec &dpdt, double /*t*/) const {
    copy_vector_to_region(q, lr_q);

    OscTaskArgs args{static_cast<int64_t>(N)};
    Domain launch_domain = runtime->get_index_space_domain(ctx, color_is);

    IndexLauncher launcher(
        OSC_BLOCK_TASK_ID,
        launch_domain,
        TaskArgument(&args, sizeof(args)),
        ArgumentMap());

    launcher.add_region_requirement(
        RegionRequirement(lr_q, READ_ONLY, EXCLUSIVE, lr_q));
    launcher.region_requirements[0].add_field(FID_VAL);

    launcher.add_region_requirement(
        RegionRequirement(lp_dpdt, 0 /*identity projection*/, WRITE_DISCARD,
                          EXCLUSIVE, lr_dpdt));
    launcher.region_requirements[1].add_field(FID_VAL);

    FutureMap fm = runtime->execute_index_space(ctx, launcher);
    fm.wait_all_results();

    copy_region_to_vector(lr_dpdt, dpdt);
  }

private:
  void copy_vector_to_region(const dvec &src, LogicalRegion lr) const {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
    for (coord_t i = 0; i < static_cast<coord_t>(N); ++i) {
      acc[Point<1>(i)] = src[static_cast<size_t>(i)];
    }

    runtime->unmap_region(ctx, pr);
  }

  void copy_region_to_vector(LogicalRegion lr, dvec &dst) const {
    dst.resize(N);

    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);
    PhysicalRegion pr = runtime->map_region(ctx, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VAL);
    for (coord_t i = 0; i < static_cast<coord_t>(N); ++i) {
      dst[static_cast<size_t>(i)] = acc[Point<1>(i)];
    }

    runtime->unmap_region(ctx, pr);
  }

private:
  Runtime *runtime;
  Context ctx;
  size_t N, G, M;

  IndexSpace is, color_is;
  FieldSpace fs;
  LogicalRegion lr_q, lr_dpdt;
  IndexPartition ip_dpdt;
  LogicalPartition lp_dpdt;
};

struct SimConfig {
  size_t N = 1024;
  size_t G = 128;
  size_t steps = 100;
  double dt = 0.01;
};

SimConfig parse_config(const InputArgs &args) {
  SimConfig cfg;
  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", po::value<size_t>(&cfg.N)->default_value(1024), "Dimension (1024)")
      ("G", po::value<size_t>(&cfg.G)->default_value(128), "Block size (128)")
      ("steps", po::value<size_t>(&cfg.steps)->default_value(100), "time steps (100)")
      ("dt", po::value<double>(&cfg.dt)->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(desc)
                    .allow_unregistered()
                    .run();
  po::store(parsed, vm);
  po::notify(vm);

  return cfg;
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx,
                    Runtime *runtime) {
  const InputArgs &args = Runtime::get_input_args();
  const SimConfig cfg = parse_config(args);

  if (cfg.N == 0 || cfg.G == 0 || (cfg.N % cfg.G) != 0) {
    std::cerr << "Invalid configuration: require N > 0, G > 0, N % G == 0.\n";
    return;
  }

  const size_t N = cfg.N;
  const size_t G = cfg.G;
  const size_t M = N / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << cfg.steps
          << ", dt: " << cfg.dt << std::endl;

  dvec q(N, 0.0), p(N);
  std::mt19937 engine(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &v : p) v = dist(engine);

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p))) << std::endl;

  LegionOscChain osc(runtime, ctx, N, G);
  using stepper_type = odeint::symplectic_rkn_sb3a_mclachlan<dvec>;

  stepper_type stepper;
  double t = 0.0;
  auto state = std::make_pair(boost::ref(q), boost::ref(p));
  for (size_t step = 0; step < cfg.steps; ++step) {
    stepper.do_step(boost::ref(osc), state, t, cfg.dt);
    t += cfg.dt;
  }

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p))) << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(OSC_BLOCK_TASK_ID, "osc_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<osc_block_task>(registrar, "osc_block");
  }

  return Runtime::start(argc, argv);
}
