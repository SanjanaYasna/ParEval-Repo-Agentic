#include "legion.h"

#include <boost/program_options.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;

namespace {

constexpr double KAPPA  = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return std::copysign(std::pow(std::abs(x), k), x);
}

double compute_energy(const std::vector<double>& q, const std::vector<double>& p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.0;

  e += 0.5 * checked_pow(q[0], LAMBDA) / LAMBDA;

  for (std::size_t i = 0; i + 1 < N; ++i) {
    e += 0.5 * p[i] * p[i]
       + checked_pow(q[i], KAPPA) / KAPPA
       + checked_pow(q[i] - q[i + 1], LAMBDA) / LAMBDA;
  }

  e += 0.5 * p[N - 1] * p[N - 1]
     + checked_pow(q[N - 1], KAPPA) / KAPPA
     + 0.5 * checked_pow(q[N - 1], LAMBDA) / LAMBDA;

  return e;
}

struct coor_deriv {
  void operator()(const std::vector<double>& p, std::vector<double>& dqdt) const {
    dqdt = p;
  }
};

struct momentum_deriv {
  void operator()(const std::vector<double>& q, std::vector<double>& dpdt) const {
    const std::size_t N = q.size();
    dpdt.resize(N);
    if (N == 0) return;

    for (std::size_t i = 0; i < N; ++i) {
      const double qi = q[i];

      const double left =
          (i == 0) ? -signed_pow(qi, LAMBDA - 1.0)
                   : signed_pow(q[i - 1] - qi, LAMBDA - 1.0);

      const double right =
          (i + 1 == N) ? signed_pow(qi, LAMBDA - 1.0)
                       : signed_pow(qi - q[i + 1], LAMBDA - 1.0);

      dpdt[i] = -signed_pow(qi, KAPPA - 1.0) + left - right;
    }
  }
};

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
  namespace po = boost::program_options;
  namespace odeint = boost::numeric::odeint;

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

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << steps
          << ", dt: " << dt << std::endl;

  std::vector<double> q(N, 0.0), p(N);

  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::mt19937 engine(0);
  for (std::size_t i = 0; i < N; ++i) {
    p[i] = distribution(engine);
  }

  const long long e_init =
      static_cast<long long>(std::llround(compute_energy(q, p)));
  outfile << "Initialization complete, energy: " << e_init << std::endl;

  using stepper_type = odeint::symplectic_rkn_sb3a_mclachlan<std::vector<double>>;
  stepper_type stepper;
  auto system = std::make_pair(coor_deriv{}, momentum_deriv{});

  double t = 0.0;
  for (std::size_t s = 0; s < steps; ++s) {
    stepper.do_step(system, q, p, t, dt);
    t += dt;
  }

  const double e_final_val = compute_energy(q, p);
  const long long e_final =
      static_cast<long long>(std::llround(e_final_val));
  outfile << "Integration complete, energy: " << e_final << std::endl;
}

} // end anonymous namespace

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

  return Runtime::start(argc, argv);
}
