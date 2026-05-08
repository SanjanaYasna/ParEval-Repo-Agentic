// Translated from HPX to Legion execution model
#include "legion.h"

#include <boost/numeric/odeint/integrate/integrate_n_steps.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/program_options.hpp>

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
namespace po = boost::program_options;
using boost::numeric::odeint::integrate_n_steps;
using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan;

namespace {
  enum TaskIDs {
    TOP_LEVEL_TASK_ID = 1,
    DERIV_BLOCK_TASK_ID = 2
  };

  using state_type = std::vector<double>;

  constexpr double KAPPA = 3.5;
  constexpr double LAMBDA = 4.5;

  // Globals used by the ODE system functor to launch Legion tasks.
  Runtime *g_runtime = nullptr;
  Context g_context = Context();
  std::size_t g_block_size = 128;

  inline double checked_pow(double x, double y) {
    if (x == 0.0) return 0.0;
    return std::pow(std::abs(x), y);
  }

  inline double signed_pow(double x, double k) {
    if (x == 0.0) return 0.0;
    return checked_pow(x, k) * (x > 0.0 ? 1.0 : -1.0);
  }

  struct DerivTaskArgs {
    const double *q;
    double *dpdt;
    std::size_t N;
    std::size_t G;
  };

  // Computes dpdt over one block [block*G, min((block+1)*G, N)).
  void deriv_block_task(
      const Task *task,
      const std::vector<PhysicalRegion> & /*regions*/,
      Context /*ctx*/,
      Runtime * /*runtime*/) {
    const std::size_t block = static_cast<std::size_t>(task->index_point[0]);

    const auto *args = static_cast<const DerivTaskArgs *>(task->args);
    const std::size_t start = block * args->G;
    const std::size_t end = std::min(start + args->G, args->N);

    const double *q = args->q;
    double *dpdt = args->dpdt;
    const std::size_t N = args->N;

    for (std::size_t i = start; i < end; ++i) {
      const double left = (i == 0) ? 0.0 : q[i - 1];
      const double right = (i + 1 == N) ? 0.0 : q[i + 1];

      const double onsite = signed_pow(q[i], KAPPA - 1.0);
      const double coupling_left = -signed_pow(q[i] - left, LAMBDA - 1.0);
      const double coupling_right = signed_pow(q[i] - right, LAMBDA - 1.0);

      dpdt[i] = -onsite + coupling_left - coupling_right;
    }
  }

  struct osc_chain_system {
    void operator()(const state_type &q, state_type &dpdt) const {
      const std::size_t N = q.size();
      dpdt.resize(N);
      if (N == 0) return;

      const std::size_t num_blocks = (N + g_block_size - 1) / g_block_size;
      DerivTaskArgs args{q.data(), dpdt.data(), N, g_block_size};

      Rect<1> launch_rect(
          Point<1>(0),
          Point<1>(static_cast<coord_t>(num_blocks - 1)));

      IndexLauncher launcher(
          DERIV_BLOCK_TASK_ID,
          Domain(launch_rect),
          TaskArgument(&args, sizeof(args)),
          ArgumentMap());

      FutureMap fm = g_runtime->execute_index_space(g_context, launcher);
      fm.wait_all_results();
    }
  };

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

  using stepper_type = symplectic_rkn_sb3a_mclachlan<state_type>;

  void top_level_task(
      const Task * /*task*/,
      const std::vector<PhysicalRegion> & /*regions*/,
      Context ctx,
      Runtime *runtime) {
    g_runtime = runtime;
    g_context = ctx;

    const InputArgs &input_args = Runtime::get_input_args();

    po::options_description desc_commandline("Usage: odeint [options]");
    desc_commandline.add_options()
      ("help,h", "Print help")
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

    po::variables_map vm;
    try {
      auto parsed = po::command_line_parser(input_args.argc, input_args.argv)
                        .options(desc_commandline)
                        .allow_unregistered() // Ignore Legion runtime flags.
                        .run();
      po::store(parsed, vm);
      po::notify(vm);
    } catch (const std::exception &e) {
      std::cerr << "Error parsing command line: " << e.what() << "\n";
      return;
    }

    if (vm.count("help")) {
      std::cout << desc_commandline << "\n";
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

    g_block_size = G;
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

    state_type q_in(N, 0.0);
    state_type p_in(N);

    std::mt19937 engine(0);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
      p_in[i] = distribution(engine);
    }

    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::llround(energy(q_in, p_in)))
            << std::endl;

    integrate_n_steps(
        stepper_type(),
        osc_chain_system(),
        std::make_pair(std::ref(q_in), std::ref(p_in)),
        0.0,
        dt,
        steps);

    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::llround(energy(q_in, p_in)))
            << std::endl;
  }
} // namespace

int main(int argc, char **argv) {
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(DERIV_BLOCK_TASK_ID, "deriv_block");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<deriv_block_task>(registrar, "deriv_block");
  }

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  return Runtime::start(argc, argv);
}
