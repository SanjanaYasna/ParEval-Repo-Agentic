#include <legion.h>
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

namespace {

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

enum TaskIDs : TaskID {
    TOP_LEVEL_TASK_ID = 1
};

inline double checked_pow(double x, double y) {
    if (x == 0.0) return 0.0;
    return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
    if (x == 0.0) return 0.0;
    return std::copysign(checked_pow(x, k), x);
}

struct osc_chain_momentum {
    void operator()(const std::vector<double>& q, std::vector<double>& dpdt) const {
        const std::size_t n = q.size();
        dpdt.resize(n);

        for (std::size_t i = 0; i < n; ++i) {
            const double qi = q[i];

            const double left_term =
                (i == 0) ? signed_pow(qi, LAMBDA - 1.0)
                         : signed_pow(qi - q[i - 1], LAMBDA - 1.0);

            const double right_term =
                (i + 1 == n) ? signed_pow(qi, LAMBDA - 1.0)
                             : signed_pow(qi - q[i + 1], LAMBDA - 1.0);

            dpdt[i] = -signed_pow(qi, KAPPA - 1.0) - left_term - right_term;
        }
    }
};

double compute_energy(const std::vector<double>& q, const std::vector<double>& p) {
    const std::size_t n = q.size();
    if (n == 0) return 0.0;

    double e = 0.5 * checked_pow(q[0], LAMBDA) / LAMBDA;

    for (std::size_t i = 0; i + 1 < n; ++i) {
        e += 0.5 * p[i] * p[i]
           + checked_pow(q[i], KAPPA) / KAPPA
           + checked_pow(q[i] - q[i + 1], LAMBDA) / LAMBDA;
    }

    e += 0.5 * p[n - 1] * p[n - 1]
       + checked_pow(q[n - 1], KAPPA) / KAPPA
       + 0.5 * checked_pow(q[n - 1], LAMBDA) / LAMBDA;

    return e;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*) {
    const InputArgs& in = Runtime::get_input_args();
    boost::program_options::variables_map vm;
    boost::program_options::options_description desc("Usage: odeint [options]");

    desc.add_options()
        ("N", boost::program_options::value<std::size_t>()->default_value(1024), "Dimension (1024)")
        ("G", boost::program_options::value<std::size_t>()->default_value(128), "Block size (128)")
        ("steps", boost::program_options::value<std::size_t>()->default_value(100), "time steps (100)")
        ("dt", boost::program_options::value<double>()->default_value(0.01), "step size (0.01)");

    auto parsed = boost::program_options::command_line_parser(in.argc, in.argv)
                    .options(desc)
                    .allow_unregistered()
                    .run();
    boost::program_options::store(parsed, vm);
    boost::program_options::notify(vm);

    const std::size_t N_in = vm["N"].as<std::size_t>();
    const std::size_t G = vm["G"].as<std::size_t>();
    const std::size_t steps = vm["steps"].as<std::size_t>();
    const double dt = vm["dt"].as<double>();

    if (G == 0) {
        std::cerr << "Invalid argument: G must be > 0\n";
        return;
    }

    const std::size_t M = N_in / G;
    const std::size_t N = M * G;

    if (M == 0 || N == 0) {
        std::cerr << "Invalid configuration: N/G must be >= 1\n";
        return;
    }

    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open odeint.txt for writing.\n";
        return;
    }

    outfile << "Dimension: " << N_in
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M
            << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    std::vector<double> q(N, 0.0);
    std::vector<double> p(N, 0.0);

    {
        std::mt19937 engine(0);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (std::size_t i = 0; i < N; ++i) p[i] = dist(engine);
    }

    const double e_init = compute_energy(q, p);
    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::llround(e_init)) << std::endl;

    using state_type = std::vector<double>;
    using phase_space_type = std::pair<state_type, state_type>;
    using stepper_type = boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan<state_type>;

    phase_space_type state{std::move(q), std::move(p)};
    stepper_type stepper;
    osc_chain_momentum system;

    double t = 0.0;
    for (std::size_t s = 0; s < steps; ++s) {
        stepper.do_step(system, state, t, dt);
        t += dt;
    }

    const double e_final = compute_energy(state.first, state.second);
    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::llround(e_final)) << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");

    return Runtime::start(argc, argv);
}
