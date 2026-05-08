// Copyright 2013 Mario Mulansky
// Legion port of odeint.cpp

#include <legion.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <functional>
#include <utility>
#include <algorithm>

#include <boost/program_options.hpp>
#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp>
#include <boost/ref.hpp>

#include "shared_resize.hpp"
#include "algebra.hpp"
#include "shared_operations.hpp"
#include "system.hpp"

using namespace Legion;
namespace po = boost::program_options;
using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan;

enum TaskIDs : TaskID
{
    TOP_LEVEL_TASK_ID = 1
};

struct initialize_zero
{
    const size_t m_N;

    explicit initialize_zero(const size_t N) : m_N(N) {}

    shared_vec operator()(shared_vec v) const
    {
        v->resize(m_N);
        std::fill(v->begin(), v->end(), 0.0);
        return v;
    }
};

struct initialize_copy
{
    const dvec &m_data;
    const size_t m_index;
    const size_t m_len;

    initialize_copy(const dvec &data, const size_t index, const size_t len)
        : m_data(data), m_index(index), m_len(len) {}

    shared_vec operator()(shared_vec v) const
    {
        v->resize(m_len);
        std::copy(&(m_data[m_index]), &(m_data[m_index + m_len]), v->begin());
        return v;
    }
};

typedef symplectic_rkn_sb3a_mclachlan<state_type,
                                      state_type,
                                      double,
                                      state_type,
                                      state_type,
                                      double,
                                      local_dataflow_algebra,
                                      local_dataflow_shared_operations>
    stepper_type;

static bool parse_command_line(const InputArgs &args, po::variables_map &vm)
{
    po::options_description desc_commandline("Usage: odeint [options] [Legion runtime options]");
    desc_commandline.add_options()
        ("help,h", "Print help")
        ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
        ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
        ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
        ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

    auto parsed = po::command_line_parser(args.argc, args.argv)
                      .options(desc_commandline)
                      .allow_unregistered() // allow Legion flags like -ll:cpu
                      .run();

    po::store(parsed, vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc_commandline << std::endl;
        std::cout << "Example (parallel Legion run): ./odeint --N 2048 --dt 0.1 -ll:cpu 4" << std::endl;
        return false;
    }

    return true;
}

void top_level_task(const Task *,
                    const std::vector<PhysicalRegion> &,
                    Context ctx,
                    Runtime *runtime)
{
    bind_system_runtime(ctx, runtime);

    po::variables_map vm;
    if (!parse_command_line(Runtime::get_input_args(), vm))
        return;

    const std::size_t N = vm["N"].as<std::size_t>();
    const std::size_t G = vm["G"].as<std::size_t>();
    const std::size_t steps = vm["steps"].as<std::size_t>();
    const double dt = vm["dt"].as<double>();

    if (G == 0 || (N % G) != 0)
    {
        std::cerr << "Invalid configuration: require G > 0 and N % G == 0. Got N="
                  << N << ", G=" << G << std::endl;
        return;
    }

    const std::size_t M = N / G;

    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open())
    {
        std::cerr << "Failed to open odeint.txt for writing." << std::endl;
        return;
    }

    outfile << "Dimension: " << N << ", number of elements per dataflow: " << G
            << ", number of dataflow: " << M << ", steps: " << steps << ", dt: " << dt << std::endl;

    dvec p_init(N);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::mt19937 engine(0);
    auto generator = std::bind(distribution, engine);
    std::generate(p_init.begin(), p_init.end(), std::ref(generator));

    state_type q_in(M);
    state_type p_in(M);

    for (size_t i = 0; i < M; ++i)
    {
        q_in[i] = initialize_zero(G)(std::make_shared<dvec>());
        p_in[i] = initialize_copy(p_init, i * G, G)(std::make_shared<dvec>());
    }

    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(energy(q_in, p_in))) << std::endl;

    stepper_type stepper;
    auto state = std::make_pair(boost::ref(q_in), boost::ref(p_in));
    double t = 0.0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        stepper.do_step(osc_chain, state, t, dt);
        t += dt;
    }

    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(energy(q_in, p_in))) << std::endl;
}

int main(int argc, char **argv)
{
    register_system_tasks();

    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    return Runtime::start(argc, argv);
}
