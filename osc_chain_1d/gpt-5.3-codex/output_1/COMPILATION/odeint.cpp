#include "legion.h"

#include <boost/program_options.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;

namespace po = boost::program_options;
namespace odeint = boost::numeric::odeint;

namespace {

constexpr double KAPPA  = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

using vec_type = std::vector<double>;
using phase_state = std::pair<vec_type, vec_type>;

inline double abs_pow(const double x, const double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(const double x, const double y) {
  if (x == 0.0) return 0.0;
  return std::copysign(std::pow(std::abs(x), y), x);
}

struct coord_deriv {
  void operator()(const vec_type& p, vec_type& dqdt) const {
    dqdt = p;
  }
};

struct momentum_deriv {
  void operator()(const vec_type& q, vec_type& dpdt) const {
    const std::size_t n = q.size();
    dpdt.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
      const double qi = q[i];
      const double ql = (i == 0) ? 0.0 : q[i - 1];
      const double qr = (i + 1 == n) ? 0.0 : q[i + 1];

      const double coupling_left  = signed_pow(ql - qi, LAMBDA - 1.0);
      const double coupling_right = signed_pow(qi - qr, LAMBDA - 1.0);

      dpdt[i] = -signed_pow(qi, KAPPA - 1.0) + coupling_left - coupling_right;
    }
  }
};

double compute_total_energy(const vec_type& q, const vec_type& p) {
  const std::size_t n = q.size();
  double e = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const double qi = q[i];
    const double pi = p[i];

    e += 0.5 * pi * pi;
    e += abs_pow(qi, KAPPA) / KAPPA;

    if (i == 0) {
      e += 0.5 * abs_pow(qi, LAMBDA) / LAMBDA;
    }
    if (i + 1 < n) {
      e += abs_pow(qi - q[i + 1], LAMBDA) / LAMBDA;
    }
    if (i + 1 == n) {
      e += 0.5 * abs_pow(qi, LAMBDA) / LAMBDA;
    }
  }

  return e;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
  const InputArgs& command_args = Runtime::get_input_args();

  po::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)")
      ("help", "Print help");

  po::variables_map vm;
  try {
    auto parsed = po::command_line_parser(command_args.argc, command_args.argv)
                      .options(desc)
                      .allow_unregistered()
                      .run();
    po::store(parsed, vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Argument parsing error: " << e.what() << "\n";
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
    std::cerr << "N and G must be > 0.\n";
    return;
  }
  if (N % G != 0) {
    std::cerr << "For this Legion translation, N must be divisible by G.\n";
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
          << ", dt: " << dt << "\n";

  vec_type q(N, 0.0);
  vec_type p(N, 0.0);

  std::mt19937 engine(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < N; ++i) {
    p[i] = dist(engine);
  }

  const double e_init = compute_total_energy(q, p);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::round(e_init)) << "\n";

  phase_state x(q, p);
  odeint::symplectic_rkn_sb3a_mclachlan<vec_type> stepper;
  auto system = std::make_pair(coord_deriv{}, momentum_deriv{});

  double t = 0.0;
  for (std::size_t s = 0; s < steps; ++s) {
    stepper.do_step(system, x, t, dt);
    t += dt;
  }

  const double e_final = compute_total_energy(x.first, x.second);
  outfile << "Integration complete, energy: "
          << static_cast<long long>(std::round(e_final)) << "\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

  return Runtime::start(argc, argv);
}
