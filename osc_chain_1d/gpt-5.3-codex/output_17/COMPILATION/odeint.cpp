#include "legion.h"

#include <boost/numeric/odeint.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace Legion;
using namespace boost::numeric::odeint;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct AppConfig {
  std::size_t N = 1024;
  std::size_t G = 128;
  std::size_t steps = 100;
  double dt = 0.01;
  bool help = false;
};

inline double checked_pow_abs(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return checked_pow_abs(x, k) * (x > 0.0 ? 1.0 : -1.0);
}

static void print_usage() {
  std::cout
      << "Usage: ./odeint [options] [Legion options]\n"
      << "Options:\n"
      << "  --N <int>       Dimension (default 1024)\n"
      << "  --G <int>       Block size (default 128)\n"
      << "  --steps <int>   Time steps (default 100)\n"
      << "  --dt <double>   Step size (default 0.01)\n"
      << "  --help          Show this message\n"
      << "\nExample:\n"
      << "  ./odeint --N 2048 --dt 0.1 -ll:cpu 4\n";
}

static bool parse_value_arg(const std::string &arg, const char *name, std::string &value_out) {
  const std::string prefix = std::string(name) + "=";
  if (arg.rfind(prefix, 0) == 0) {
    value_out = arg.substr(prefix.size());
    return true;
  }
  return false;
}

static bool parse_args(const InputArgs &input, AppConfig &cfg) {
  for (int i = 1; i < input.argc; ++i) {
    std::string a(input.argv[i]);
    if (a == "--help") {
      cfg.help = true;
      return true;
    }

    std::string v;
    if (a == "--N" || parse_value_arg(a, "--N", v)) {
      if (a == "--N") {
        if (i + 1 >= input.argc) return false;
        v = input.argv[++i];
      }
      cfg.N = static_cast<std::size_t>(std::stoull(v));
      continue;
    }
    if (a == "--G" || parse_value_arg(a, "--G", v)) {
      if (a == "--G") {
        if (i + 1 >= input.argc) return false;
        v = input.argv[++i];
      }
      cfg.G = static_cast<std::size_t>(std::stoull(v));
      continue;
    }
    if (a == "--steps" || parse_value_arg(a, "--steps", v)) {
      if (a == "--steps") {
        if (i + 1 >= input.argc) return false;
        v = input.argv[++i];
      }
      cfg.steps = static_cast<std::size_t>(std::stoull(v));
      continue;
    }
    if (a == "--dt" || parse_value_arg(a, "--dt", v)) {
      if (a == "--dt") {
        if (i + 1 >= input.argc) return false;
        v = input.argv[++i];
      }
      cfg.dt = std::stod(v);
      continue;
    }
    // Ignore unknown args (e.g. Legion runtime args like -ll:cpu 4)
  }
  return true;
}

static double compute_energy(const std::vector<double> &q, const std::vector<double> &p) {
  const std::size_t N = q.size();
  if (N == 0) return 0.0;

  double e = 0.5 * checked_pow_abs(q[0], LAMBDA) / LAMBDA;
  for (std::size_t i = 0; i + 1 < N; ++i) {
    e += 0.5 * p[i] * p[i]
       + checked_pow_abs(q[i], KAPPA) / KAPPA
       + checked_pow_abs(q[i] - q[i + 1], LAMBDA) / LAMBDA;
  }
  e += 0.5 * p[N - 1] * p[N - 1]
     + checked_pow_abs(q[N - 1], KAPPA) / KAPPA
     + 0.5 * checked_pow_abs(q[N - 1], LAMBDA) / LAMBDA;
  return e;
}

struct osc_chain_force {
  void operator()(const std::vector<double> &q, std::vector<double> &dpdt) const {
    const std::size_t N = q.size();
    dpdt.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
      const double qi = q[i];
      const double ql = (i == 0) ? 0.0 : q[i - 1];
      const double qr = (i + 1 == N) ? 0.0 : q[i + 1];

      const double onsite = -signed_pow(qi, KAPPA - 1.0);
      const double left_coupling = signed_pow(ql - qi, LAMBDA - 1.0);
      const double right_coupling = -signed_pow(qi - qr, LAMBDA - 1.0);
      dpdt[i] = onsite + left_coupling + right_coupling;
    }
  }

  void operator()(const std::vector<double> &q, std::vector<double> &dpdt, const double /*t*/) const {
    (*this)(q, dpdt);
  }
};

struct osc_chain_flow {
  void operator()(const std::vector<double> &p, std::vector<double> &dqdt) const {
    dqdt = p;
  }

  void operator()(const std::vector<double> &p, std::vector<double> &dqdt, const double /*t*/) const {
    dqdt = p;
  }
};

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
  AppConfig cfg;
  const InputArgs &input = Runtime::get_input_args();
  if (!parse_args(input, cfg)) {
    std::cerr << "Invalid command line arguments.\n";
    print_usage();
    return;
  }
  if (cfg.help) {
    print_usage();
    return;
  }

  if (cfg.N == 0 || cfg.G == 0 || (cfg.N % cfg.G) != 0) {
    std::cerr << "Error: require N > 0, G > 0, and N % G == 0.\n";
    return;
  }

  const std::size_t M = cfg.N / cfg.G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing.\n";
    return;
  }

  outfile << "Dimension: " << cfg.N
          << ", number of elements per dataflow: " << cfg.G
          << ", number of dataflow: " << M
          << ", steps: " << cfg.steps
          << ", dt: " << cfg.dt << std::endl;

  std::vector<double> q(cfg.N, 0.0), p(cfg.N, 0.0);
  std::mt19937 engine(0);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (std::size_t i = 0; i < cfg.N; ++i) {
    p[i] = distribution(engine);
  }

  const long long e_init = static_cast<long long>(std::llround(compute_energy(q, p)));
  outfile << "Initialization complete, energy: " << e_init << std::endl;

  using state_type = std::vector<double>;
  using phase_type = std::pair<state_type, state_type>;
  phase_type state(q, p);

  symplectic_rkn_sb3a_mclachlan<state_type> stepper;
  auto system = std::make_pair(osc_chain_force(), osc_chain_flow());

  double t = 0.0;
  for (std::size_t step = 0; step < cfg.steps; ++step) {
    stepper.do_step(system, state, t, cfg.dt);
    t += cfg.dt;
  }

  const long long e_final = static_cast<long long>(std::llround(compute_energy(state.first, state.second)));
  outfile << "Integration complete, energy: " << e_final << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

  return Runtime::start(argc, argv);
}
