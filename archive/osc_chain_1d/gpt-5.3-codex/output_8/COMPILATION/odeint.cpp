// Copyright 2013 Mario Mulansky
// Legion translation of odeint.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>
#include <cstddef>

#include "legion.h"

#include <boost/serialization/array.hpp>
#include <boost/program_options.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/ref.hpp>

#include "shared_resize.hpp"
#include "algebra.hpp"
#include "shared_operations.hpp"
#include "system.hpp"

using namespace Legion;
using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan;

typedef symplectic_rkn_sb3a_mclachlan<
    state_type,
    state_type,
    double,
    state_type,
    state_type,
    double,
    local_dataflow_algebra,
    local_dataflow_shared_operations
> stepper_type;

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 1,
    INIT_Q_BLOCK_TASK_ID,
    INIT_P_BLOCK_TASK_ID
};

// Shared process-local state used by init tasks.
static state_type* g_q_blocks = nullptr;
static state_type* g_p_blocks = nullptr;
static dvec* g_p_init = nullptr;
static std::size_t g_block_size = 0;

static void init_q_block_task(
    const Task* task,
    const std::vector<PhysicalRegion>&,
    Context,
    Runtime*)
{
    const std::size_t i = static_cast<std::size_t>(task->index_point[0]);
    shared_vec& block = (*g_q_blocks)[i];
    if (!block) block = std::make_shared<dvec>();
    block->assign(g_block_size, 0.0);
}

static void init_p_block_task(
    const Task* task,
    const std::vector<PhysicalRegion>&,
    Context,
    Runtime*)
{
    const std::size_t i = static_cast<std::size_t>(task->index_point[0]);
    shared_vec& block = (*g_p_blocks)[i];
    if (!block) block = std::make_shared<dvec>();
    block->resize(g_block_size);

    const std::size_t offset = i * g_block_size;
    std::copy_n(g_p_init->begin() + offset, g_block_size, block->begin());
}

static bool parse_cli(
    const InputArgs& args,
    boost::program_options::variables_map& vm)
{
    namespace po = boost::program_options;

    po::options_description desc_commandline("Usage: odeint [options]");
    desc_commandline.add_options()
        ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
        ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
        ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
        ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

    auto parsed = po::command_line_parser(args.argc, args.argv)
                      .options(desc_commandline)
                      .allow_unregistered()
                      .run();

    po::store(parsed, vm);
    po::notify(vm);
    return true;
}

static void top_level_task(
    const Task*,
    const std::vector<PhysicalRegion>&,
    Context ctx,
    Runtime* runtime)
{
    boost::program_options::variables_map vm;
    if (!parse_cli(Runtime::get_input_args(), vm)) {
        std::cerr << "Failed to parse command line options.\n";
        return;
    }

    const std::size_t N = vm["N"].as<std::size_t>();
    const std::size_t G = vm["G"].as<std::size_t>();
    const std::size_t steps = vm["steps"].as<std::size_t>();
    const double dt = vm["dt"].as<double>();

    if (G == 0 || (N % G) != 0) {
        std::cerr << "Invalid arguments: G must be > 0 and N must be divisible by G.\n";
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
            << ", dt: " << dt << std::endl;

    dvec p_init(N);
    std::mt19937 engine(0);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
        p_init[i] = distribution(engine);
    }

    state_type q_in(M), p_in(M);

    g_q_blocks = &q_in;
    g_p_blocks = &p_in;
    g_p_init = &p_init;
    g_block_size = G;

    if (M > 0) {
        Rect<1> launch_rect(0, static_cast<coord_t>(M - 1));
        Domain launch_domain(launch_rect);
        ArgumentMap arg_map;

        IndexLauncher q_launcher(
            INIT_Q_BLOCK_TASK_ID, launch_domain, TaskArgument(nullptr, 0), arg_map);
        FutureMap q_fm = runtime->execute_index_space(ctx, q_launcher);
        q_fm.wait_all_results();

        IndexLauncher p_launcher(
            INIT_P_BLOCK_TASK_ID, launch_domain, TaskArgument(nullptr, 0), arg_map);
        FutureMap p_fm = runtime->execute_index_space(ctx, p_launcher);
        p_fm.wait_all_results();
    }

    g_q_blocks = nullptr;
    g_p_blocks = nullptr;
    g_p_init = nullptr;
    g_block_size = 0;

    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::round(energy(q_in, p_in))) << std::endl;

    boost::numeric::odeint::integrate_n_steps(
        stepper_type(),
        osc_chain,
        std::make_pair(boost::ref(q_in), boost::ref(p_in)),
        0.0, dt, steps);

    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::round(energy(q_in, p_in))) << std::endl;
}

int main(int argc, char* argv[])
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(INIT_Q_BLOCK_TASK_ID, "init_q_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<init_q_block_task>(registrar, "init_q_block");
    }
    {
        TaskVariantRegistrar registrar(INIT_P_BLOCK_TASK_ID, "init_p_block");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf();
        Runtime::preregister_task_variant<init_p_block_task>(registrar, "init_p_block");
    }

    return Runtime::start(argc, argv);
}
