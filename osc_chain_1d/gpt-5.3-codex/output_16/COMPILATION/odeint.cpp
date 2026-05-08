// Translated from HPX to Legion runtime model.
// The numerical method remains boost::numeric::odeint symplectic RKN integration.

#include "legion.h"

#include <boost/serialization/array.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/program_options.hpp>
#include <boost/ref.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Legion;

enum TaskIDs
{
    TOP_LEVEL_TASK_ID = 1
};

constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

inline double checked_pow(double x, double y)
{
    if (x == 0.0) return 0.0;
    return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k)
{
    const double s = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
    return checked_pow(x, k) * s;
}

using state_type = std::vector<double>;

struct osc_chain_system
{
    static void compute(const state_type &q, state_type &dpdt)
    {
        const std::size_t N = q.size();
        if (N == 0) {
            dpdt.clear();
            return;
        }

        dpdt.resize(N);

        double coupling_lr = -signed_pow(q[0], LAMBDA - 1.0);
        for (std::size_t i = 0; i + 1 < N; ++i)
        {
            dpdt[i] = -signed_pow(q[i], KAPPA - 1.0) + coupling_lr;
            coupling_lr = signed_pow(q[i] - q[i + 1], LAMBDA - 1.0);
            dpdt[i] -= coupling_lr;
        }

        dpdt[N - 1] = -signed_pow(q[N - 1], KAPPA - 1.0)
                    + coupling_lr
                    - signed_pow(q[N - 1], LAMBDA - 1.0);
    }

    void operator()(const state_type &q, state_type &dpdt) const
    {
        compute(q, dpdt);
    }

    void operator()(const state_type &q, state_type &dpdt, const double /*t*/) const
    {
        compute(q, dpdt);
    }
};

double energy(const state_type &q, const state_type &p)
{
    const std::size_t N = q.size();
    if (N == 0) return 0.0;

    double e = 0.5 * checked_pow(std::abs(q[0]), LAMBDA) / LAMBDA;
    for (std::size_t i = 0; i + 1 < N; ++i)
    {
        e += 0.5 * p[i] * p[i]
           + checked_pow(q[i], KAPPA) / KAPPA
           + checked_pow(std::abs(q[i] - q[i + 1]), LAMBDA) / LAMBDA;
    }
    e += 0.5 * p[N - 1] * p[N - 1]
       + checked_pow(q[N - 1], KAPPA) / KAPPA
       + 0.5 * checked_pow(std::abs(q[N - 1]), LAMBDA) / LAMBDA;
    return e;
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context,
                    Runtime*)
{
    namespace po = boost::program_options;
    using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan;

    // Defaults (same as original HPX code)
    std::size_t N = 1024;
    std::size_t G = 128;
    std::size_t steps = 100;
    double dt = 0.01;

    try
    {
        const InputArgs &args = Runtime::get_input_args();

        po::options_description desc_commandline("Usage: odeint [options]");
        desc_commandline.add_options()
            ("help", "Print help")
            ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
            ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
            ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
            ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

        po::variables_map vm;
        auto parsed = po::command_line_parser(args.argc, args.argv)
                        .options(desc_commandline)
                        .allow_unregistered() // keep Legion runtime flags untouched
                        .run();
        po::store(parsed, vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc_commandline << "\n";
            std::cout << "Legion parallel run example:\n"
                      << "  ./odeint --N 2048 --dt 0.1 -ll:cpu 8\n";
            return;
        }

        N = vm["N"].as<std::size_t>();
        G = vm["G"].as<std::size_t>();
        steps = vm["steps"].as<std::size_t>();
        dt = vm["dt"].as<double>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Argument parsing error: " << e.what() << std::endl;
        return;
    }

    if (N == 0 || G == 0)
    {
        std::cerr << "N and G must be > 0." << std::endl;
        return;
    }

    if (N % G != 0)
    {
        std::cerr << "Error: N must be divisible by G (got N=" << N
                  << ", G=" << G << ")." << std::endl;
        return;
    }

    const std::size_t M = N / G;

    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N
            << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M
            << ", steps: " << steps
            << ", dt: " << dt << std::endl;

    // Initialize q and p
    state_type q(N, 0.0);
    state_type p(N, 0.0);

    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    for (std::size_t i = 0; i < N; ++i)
        p[i] = distribution(engine);

    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::llround(energy(q, p))) << std::endl;

    using stepper_type = symplectic_rkn_sb3a_mclachlan<state_type>;
    stepper_type stepper;
    auto state = std::make_pair(boost::ref(q), boost::ref(p));

    double t = 0.0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        stepper.do_step(osc_chain_system(), state, t, dt);
        t += dt;
    }

    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::llround(energy(q, p))) << std::endl;
}

int main(int argc, char** argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    return Runtime::start(argc, argv);
}
