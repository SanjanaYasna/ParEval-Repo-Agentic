#include "legion.h"

#include <boost/program_options.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace Legion;
namespace odeint = boost::numeric::odeint;

namespace {

constexpr double KAPPA  = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

inline double abs_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::copysign(std::pow(std::abs(x), y), x);
}

using state_t = std::vector<double>;

struct dqdt_func {
  void operator()(const state_t& p, state_t& dqdt) const {
    dqdt = p;
  }

  template <typename Time>
  void operator()(const state_t& p, state_t& dqdt, const Time&) const {
    dqdt = p;
  }
};

struct dpdt_func {
  std::size_t N{0};

  void operator()(const state_t& q, state_t& dpdt) const {
    if (dpdt.size() != N) dpdt.resize(N);

    for (std::size_t i = 0; i < N; ++i) {
      const double qi = q[i];
      const double ql = (i == 0) ? 0.0 : q[i - 1];
      const double qr = (i + 1 == N) ? 0.0 : q[i + 1];

      const double coupling_left  = signed_pow(ql - qi, LAMBDA - 1.0);
      const double coupling_right = signed_pow(qi - qr, LAMBDA - 1.0);

      dpdt[i] = -signed_pow(qi, KAPPA - 1.0) + coupling_left - coupling_right;
    }
  }

  template <typename Time>
  void operator()(const state_t& q, state_t& dpdt, const Time&) const {
    (*this)(q, dpdt);
  }
};

double total_energy(const state_t& q, const state_t& p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double qi = q[i];
    const double pi = p[i];

    e += 0.5 * pi * pi;
    e += abs_pow(qi, KAPPA) / KAPPA;

    if (i == 0) {
      e += 0.5 * abs_pow(qi, LAMBDA) / LAMBDA;
    }
    if (i + 1 < N) {
      e += abs_pow(qi - q[i + 1], LAMBDA) / LAMBDA;
    }
    if (i + 1 == N) {
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

  boost::program_options::options_description desc("Usage: odeint [options]");
  desc.add_options()
      ("N", boost::program_options::value<std::size_t>()->default_value(1024),
       "Dimension (1024)")
      ("G", boost::program_options::value<std::size_t>()->default_value(128),
       "Block size (128)")
      ("steps", boost::program_options::value<std::size_t>()->default_value(100),
       "time steps (100)")
      ("dt", boost::program_options::value<double>()->default_value(0.01),
       "step size (0.01)")
      ("help", "Print help");

  boost::program_options::variables_map vm;
  try {
    auto parsed = boost::program_options::command_line_parser(
                      command_args.argc, command_args.argv)
                      .options(desc)
                      .allow_unregistered()
                      .run();
    boost::program_options::store(parsed, vm);
    boost::program_options::notify(vm);
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

  state_t q(N, 0.0), p(N, 0.0);

  std::mt19937 engine(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < N; ++i) {
    p[i] = dist(engine);
  }

  const double e_init = total_energy(q, p);
  outfile << "Initialization complete, energy: "
          << static_cast<long long>(std::round(e_init)) << "\n";

  using stepper_t = odeint::symplectic_rkn_sb3a_mclachlan<state_t>;
  stepper_t stepper;
  auto system = std::make_pair(dqdt_func{}, dpdt_func{N});

  double t = 0.0;
  for (std::size_t s = 0; s < steps; ++s) {
    stepper.do_step(system, q, p, t, dt);
    t += dt;
  }

  const double e_final = total_energy(q, p);
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
