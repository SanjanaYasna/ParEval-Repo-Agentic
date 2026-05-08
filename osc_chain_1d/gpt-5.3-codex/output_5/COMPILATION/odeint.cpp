#include "legion.h"

#include <boost/numeric/odeint.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace Legion;
namespace odeint = boost::numeric::odeint;

enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1
};

static constexpr double KAPPA = 3.5;
static constexpr double LAMBDA = 4.5;

struct Options {
  int64_t N = 1024;
  int64_t G = 128;
  int64_t steps = 100;
  double dt = 0.01;
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

static bool starts_with(const std::string &s, const std::string &prefix) {
  return s.rfind(prefix, 0) == 0;
}

static Options parse_options(const InputArgs &args) {
  Options opt;
  for (int i = 1; i < args.argc; ++i) {
    std::string a(args.argv[i]);

    auto parse_next_i64 = [&](int64_t &dst) {
      if (i + 1 < args.argc) dst = std::stoll(args.argv[++i]);
    };
    auto parse_next_f64 = [&](double &dst) {
      if (i + 1 < args.argc) dst = std::stod(args.argv[++i]);
    };

    if (a == "--N") parse_next_i64(opt.N);
    else if (a == "--G") parse_next_i64(opt.G);
    else if (a == "--steps") parse_next_i64(opt.steps);
    else if (a == "--dt") parse_next_f64(opt.dt);
    else if (starts_with(a, "--N=")) opt.N = std::stoll(a.substr(4));
    else if (starts_with(a, "--G=")) opt.G = std::stoll(a.substr(4));
    else if (starts_with(a, "--steps=")) opt.steps = std::stoll(a.substr(8));
    else if (starts_with(a, "--dt=")) opt.dt = std::stod(a.substr(5));
  }

  if (opt.N < 1) opt.N = 1;
  if (opt.G < 1) opt.G = 1;
  if (opt.steps < 0) opt.steps = 0;
  if (opt.G > opt.N) opt.G = opt.N;
  return opt;
}

struct OscChainSystem {
  int64_t N;

  void operator()(const std::vector<double> &q, std::vector<double> &dpdt) const {
    dpdt.resize(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
      const double qi = q[static_cast<size_t>(i)];

      const double left_coupling =
          (i == 0) ? -signed_pow(qi, LAMBDA - 1.0)
                   : signed_pow(q[static_cast<size_t>(i - 1)] - qi, LAMBDA - 1.0);

      const double right_coupling =
          (i == N - 1) ? signed_pow(qi, LAMBDA - 1.0)
                       : signed_pow(qi - q[static_cast<size_t>(i + 1)], LAMBDA - 1.0);

      dpdt[static_cast<size_t>(i)] =
          -signed_pow(qi, KAPPA - 1.0) + left_coupling - right_coupling;
    }
  }
};

static double compute_energy(const std::vector<double> &q, const std::vector<double> &p) {
  const int64_t N = static_cast<int64_t>(q.size());
  if (N == 0) return 0.0;

  double e = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    const double qi = q[static_cast<size_t>(i)];
    const double pi = p[static_cast<size_t>(i)];

    e += 0.5 * pi * pi + checked_pow(qi, KAPPA) / KAPPA;

    if (i < N - 1) {
      const double qnext = q[static_cast<size_t>(i + 1)];
      e += checked_pow(std::abs(qi - qnext), LAMBDA) / LAMBDA;
    }
    if (i == 0) {
      e += 0.5 * checked_pow(std::abs(qi), LAMBDA) / LAMBDA;
    }
    if (i == N - 1) {
      e += 0.5 * checked_pow(std::abs(qi), LAMBDA) / LAMBDA;
    }
  }
  return e;
}

static long long safe_round_energy(double e) {
  if (!std::isfinite(e)) return 0LL;
  return static_cast<long long>(std::llround(e));
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
  const Options opt = parse_options(Runtime::get_input_args());
  const int64_t N = opt.N;
  const int64_t G = opt.G;
  const int64_t steps = opt.steps;
  const double dt = opt.dt;
  const int64_t M = (N + G - 1) / G;

  std::ofstream outfile("odeint.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open odeint.txt for writing." << std::endl;
    return;
  }

  outfile << "Dimension: " << N
          << ", number of elements per dataflow: " << G
          << ", number of dataflow: " << M
          << ", steps: " << steps
          << ", dt: " << dt << std::endl;

  std::vector<double> q(static_cast<size_t>(N), 0.0);
  std::vector<double> p(static_cast<size_t>(N), 0.0);

  {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    for (int64_t i = 0; i < N; ++i) {
      p[static_cast<size_t>(i)] = distribution(engine);
    }
  }

  const double e_init = compute_energy(q, p);
  outfile << "Initialization complete, energy: " << safe_round_energy(e_init) << std::endl;

  if (steps > 0) {
    OscChainSystem system{N};
    odeint::symplectic_rkn_sb3a_mclachlan<std::vector<double>> stepper;

    double t = 0.0;
    for (int64_t s = 0; s < steps; ++s) {
      stepper.do_step(system, q, p, t, dt);
      t += dt;
    }
  }

  const double e_final = compute_energy(q, p);
  outfile << "Integration complete, energy: " << safe_round_energy(e_final) << std::endl;
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

  return Runtime::start(argc, argv);
}
