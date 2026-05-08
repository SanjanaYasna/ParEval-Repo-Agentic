// Translated from HPX to Legion execution model.
// Keeps the original CLI options: --N, --G, --steps, --dt
// Example:
//   ./odeint --N 2048 --dt 0.1 -ll:cpu 4

#include "legion.h"

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/program_options.hpp>
#include <boost/ref.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;
namespace odeint = boost::numeric::odeint;
namespace po = boost::program_options;

namespace {

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

using state_type = std::vector<double>;
using stepper_type = odeint::symplectic_rkn_sb3a_mclachlan<state_type>;

struct SimulationConfig {
  std::size_t N = 1024;
  std::size_t G = 128;   // kept for CLI compatibility with original code
  std::size_t steps = 100;
  double dt = 0.01;
  bool help = false;
};

inline double abs_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return std::copysign(abs_pow(x, k), x);
}

struct osc_chain_system {
  void operator()(const state_type &q, state_type &dpdt) const {
    const std::size_t N = q.size();
    dpdt.resize(N);

    if (N == 0) return;

    double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
    for (std::size_t i = 0; i + 1 < N; ++i) {
      dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
      coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
      dpdt[i] -= coupling_lr;
    }

    dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                + coupling_lr
                - signed_pow(q[N - 1], LAMBDA - 1.0);
  }
};

double energy(const state_type &q, const state_type &p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * abs_pow(q[0], LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i + 1 < N; ++i) {
    e += 0.5 * p[i] * p[i]
       + abs_pow(q[i], KAPPA) / KAPPA
       + abs_pow(q[i] - q[i + 1], LAMBDA) / LAMBDA;
  }

  e += 0.5 * p[N - 1] * p[N - 1]
     + abs_pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * abs_pow(q[N - 1], LAMBDA) / LAMBDA;
  return e;
}

SimulationConfig parse_config(const InputArgs &args) {
  SimulationConfig cfg;

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("help", "Print help")
      ("N", po::value<std::size_t>(&cfg.N)->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>(&cfg.G)->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>(&cfg.steps)->default_value(100), "time steps (100)")
      ("dt", po::value<double>(&cfg.dt)->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(desc)
                    .allow_unregistered() // keep Legion runtime flags working
                    .run();

  po::store(parsed, vm);
  po::notify(vm);

  if (vm.count("help")) {
    cfg.help = true;
    std::cout << desc << std::endl;
  }

  return cfg;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
  const InputArgs &args = Runtime::get_input_args();
  SimulationConfig cfg = parse_config(args);

  if (cfg.help) return;

  if (cfg.G == 0) {
    std::cerr << "Error: G must be > 0.\n";
    return;
  }

  const std::size_t M = cfg.N / cfg.G; // preserved from original output semantics

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing." << std::endl;
    return;
  }

  outfile << "Dimension: " << cfg.N
          << ", number of elements per dataflow: " << cfg.G
          << ", number of dataflow: " << M
          << ", steps: " << cfg.steps
          << ", dt: " << cfg.dt
          << std::endl;

  state_type q(cfg.N, 0.0);
  state_type p(cfg.N);

  std::mt19937 engine(0);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (double &x : p) x = distribution(engine);

  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p)))
          << std::endl;

  stepper_type stepper;
  auto x = std::make_pair(boost::ref(q), boost::ref(p));
  double t = 0.0;
  for (std::size_t i = 0; i < cfg.steps; ++i) {
    stepper.do_step(osc_chain_system(), x, t, cfg.dt);
    t += cfg.dt;
  }

  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::llround(energy(q, p)))
          << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}
