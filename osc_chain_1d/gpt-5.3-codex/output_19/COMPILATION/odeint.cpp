// Translated from HPX to Legion execution model (default mapper)
// odeint.cpp

#include <legion.h>

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Legion;

namespace {

// ----------------------------- Constants -----------------------------
constexpr double KAPPA = 3.5;
constexpr double LAMBDA = 4.5;

// ----------------------------- Legion IDs ----------------------------
enum TaskIDs {
  TOP_LEVEL_TASK_ID = 1,
  COMPUTE_BLOCK_TASK_ID
};

enum FieldIDs {
  FID_VAL = 1
};

// ----------------------------- Types ---------------------------------
using state_type = std::vector<double>;
using stepper_type = boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan<state_type>;

struct BlockTaskArgs {
  std::int64_t N;
};

// ----------------------------- Math helpers --------------------------
inline double checked_pow(double x, double y) {
  if (x == 0.0) return 0.0;
  return std::pow(std::abs(x), y);
}

inline double signed_pow(double x, double k) {
  if (x == 0.0) return 0.0;
  return checked_pow(x, k) * std::copysign(1.0, x);
}

double energy(const state_type& q, const state_type& p) {
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

// ----------------------------- CLI options ---------------------------
struct Options {
  std::size_t N = 1024;
  std::size_t G = 128;
  std::size_t steps = 100;
  double dt = 0.01;
  bool help = false;
};

Options parse_options(const InputArgs& args) {
  namespace po = boost::program_options;

  Options opts;
  po::options_description desc("Usage: odeint [options]");

  desc.add_options()
      ("help,h", "Print help")
      ("N", po::value<std::size_t>()->default_value(1024), "Dimension (1024)")
      ("G", po::value<std::size_t>()->default_value(128), "Block size (128)")
      ("steps", po::value<std::size_t>()->default_value(100), "time steps (100)")
      ("dt", po::value<double>()->default_value(0.01), "step size (0.01)");

  po::variables_map vm;
  auto parsed = po::command_line_parser(args.argc, args.argv)
                    .options(desc)
                    .allow_unregistered() // allow Legion runtime flags, e.g. -ll:cpu 4
                    .run();

  po::store(parsed, vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    opts.help = true;
    return opts;
  }

  opts.N = vm["N"].as<std::size_t>();
  opts.G = vm["G"].as<std::size_t>();
  opts.steps = vm["steps"].as<std::size_t>();
  opts.dt = vm["dt"].as<double>();
  return opts;
}

// ----------------------- Legion-backed system ------------------------
class LegionOscChainSystem {
public:
  LegionOscChainSystem(Runtime* runtime, Context ctx, std::size_t N, std::size_t G)
      : runtime_(runtime), ctx_(ctx), N_(N), G_(G), M_(N / G) {
    if (N_ == 0 || G_ == 0 || (N_ % G_) != 0) {
      throw std::runtime_error("Invalid N/G: require N>0, G>0, and N % G == 0.");
    }

    is_ = runtime_->create_index_space(ctx_, Rect<1>(0, static_cast<coord_t>(N_ - 1)));
    fs_ = runtime_->create_field_space(ctx_);
    {
      FieldAllocator allocator = runtime_->create_field_allocator(ctx_, fs_);
      allocator.allocate_field(sizeof(double), FID_VAL);
    }

    q_lr_ = runtime_->create_logical_region(ctx_, is_, fs_);
    dpdt_lr_ = runtime_->create_logical_region(ctx_, is_, fs_);

    color_is_ =
        runtime_->create_index_space(ctx_, Rect<1>(0, static_cast<coord_t>(M_ - 1)));
    dpdt_ip_ = runtime_->create_equal_partition(ctx_, is_, color_is_);
    dpdt_lp_ = runtime_->get_logical_partition(ctx_, dpdt_lr_, dpdt_ip_);
    launch_domain_ = runtime_->get_index_space_domain(ctx_, color_is_);
  }

  ~LegionOscChainSystem() {
    runtime_->destroy_logical_region(ctx_, q_lr_);
    runtime_->destroy_logical_region(ctx_, dpdt_lr_);
    runtime_->destroy_index_partition(ctx_, dpdt_ip_);
    runtime_->destroy_index_space(ctx_, color_is_);
    runtime_->destroy_field_space(ctx_, fs_);
    runtime_->destroy_index_space(ctx_, is_);
  }

  void operator()(const state_type& q, state_type& dpdt) const {
    if (q.size() != N_) {
      throw std::runtime_error("State size mismatch in system operator().");
    }
    if (dpdt.size() != N_) dpdt.resize(N_);

    write_vector_to_region(q_lr_, q);
    launch_block_tasks();
    read_region_to_vector(dpdt_lr_, dpdt);
  }

private:
  void write_vector_to_region(LogicalRegion lr, const state_type& src) const {
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);

    PhysicalRegion pr = runtime_->map_region(ctx_, launcher);
    pr.wait_until_valid();

    FieldAccessor<WRITE_DISCARD, double, 1> acc(pr, FID_VAL);
    Rect<1> rect = runtime_->get_index_space_domain(ctx_, lr.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++) {
      const coord_t idx = (*it)[0];
      acc[*it] = src[static_cast<std::size_t>(idx)];
    }

    runtime_->unmap_region(ctx_, pr);
  }

  void read_region_to_vector(LogicalRegion lr, state_type& dst) const {
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(FID_VAL);
    InlineLauncher launcher(req);

    PhysicalRegion pr = runtime_->map_region(ctx_, launcher);
    pr.wait_until_valid();

    FieldAccessor<READ_ONLY, double, 1> acc(pr, FID_VAL);
    Rect<1> rect = runtime_->get_index_space_domain(ctx_, lr.get_index_space());

    for (PointInRectIterator<1> it(rect); it(); it++) {
      const coord_t idx = (*it)[0];
      dst[static_cast<std::size_t>(idx)] = acc[*it];
    }

    runtime_->unmap_region(ctx_, pr);
  }

  void launch_block_tasks() const {
    BlockTaskArgs args{static_cast<std::int64_t>(N_)};
    ArgumentMap arg_map;

    IndexLauncher launcher(
        COMPUTE_BLOCK_TASK_ID, launch_domain_, TaskArgument(&args, sizeof(args)), arg_map);

    // Read full q in each task (simple/default-mapper friendly)
    launcher.add_region_requirement(RegionRequirement(q_lr_, READ_ONLY, EXCLUSIVE, q_lr_));
    launcher.region_requirements.back().add_field(FID_VAL);

    // Write each task's own dpdt block
    launcher.add_region_requirement(
        RegionRequirement(dpdt_lp_, 0 /* identity projection */,
                          WRITE_DISCARD, EXCLUSIVE, dpdt_lr_));
    launcher.region_requirements.back().add_field(FID_VAL);

    FutureMap fm = runtime_->execute_index_space(ctx_, launcher);
    fm.wait_all_results();
  }

private:
  Runtime* runtime_;
  Context ctx_;

  std::size_t N_;
  std::size_t G_;
  std::size_t M_;

  IndexSpace is_;
  FieldSpace fs_;
  LogicalRegion q_lr_;
  LogicalRegion dpdt_lr_;

  IndexSpace color_is_;
  IndexPartition dpdt_ip_;
  LogicalPartition dpdt_lp_;
  Domain launch_domain_;
};

// ----------------------------- Legion task ---------------------------
void compute_block_task(const Task* task,
                        const std::vector<PhysicalRegion>& regions,
                        Context ctx,
                        Runtime* runtime) {
  const auto* args = static_cast<const BlockTaskArgs*>(task->args);
  const coord_t N = static_cast<coord_t>(args->N);

  FieldAccessor<READ_ONLY, double, 1> q_acc(regions[0], FID_VAL);
  FieldAccessor<WRITE_DISCARD, double, 1> dpdt_acc(regions[1], FID_VAL);

  Rect<1> out_rect =
      runtime->get_index_space_domain(ctx, regions[1].get_logical_region().get_index_space());

  for (coord_t i = out_rect.lo[0]; i <= out_rect.hi[0]; ++i) {
    const Point<1> p_i(i);
    const double qi = q_acc[p_i];

    const double q_left = (i == 0) ? 0.0 : q_acc[Point<1>(i - 1)];
    const double q_right = (i == (N - 1)) ? 0.0 : q_acc[Point<1>(i + 1)];

    dpdt_acc[p_i] =
        -signed_pow(qi, KAPPA - 1.0)
        -signed_pow(qi - q_left, LAMBDA - 1.0)
        -signed_pow(qi - q_right, LAMBDA - 1.0);
  }
}

void top_level_task(const Task*,
                    const std::vector<PhysicalRegion>&,
                    Context ctx,
                    Runtime* runtime) {
  try {
    const Options opts = parse_options(Runtime::get_input_args());
    if (opts.help) return;

    if (opts.N == 0 || opts.G == 0 || (opts.N % opts.G) != 0) {
      std::cerr << "Error: require N > 0, G > 0, and N % G == 0.\n";
      return;
    }

    const std::size_t M = opts.N / opts.G;

    std::ofstream outfile("odeint.txt");
    if (!outfile.is_open()) {
      std::cerr << "Failed to open odeint.txt for writing.\n";
      return;
    }

    outfile << "Dimension: " << opts.N
            << ", number of elements per dataflow: " << opts.G
            << ", number of dataflow: " << M
            << ", steps: " << opts.steps
            << ", dt: " << opts.dt << std::endl;

    state_type q_in(opts.N, 0.0);
    state_type p_in(opts.N);

    std::mt19937 engine(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& v : p_in) v = dist(engine);

    outfile << "Initialization complete, energy: "
            << static_cast<long long>(std::llround(energy(q_in, p_in))) << std::endl;

    LegionOscChainSystem osc_chain(runtime, ctx, opts.N, opts.G);

    boost::numeric::odeint::integrate_n_steps(
        stepper_type(),
        std::ref(osc_chain),
        std::make_pair(std::ref(q_in), std::ref(p_in)),
        0.0, opts.dt, opts.steps);

    outfile << "Integration complete, energy: "
            << static_cast<long long>(std::llround(energy(q_in, p_in))) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception in top_level_task: " << e.what() << std::endl;
  }
}

} // namespace

// ----------------------------- main ----------------------------------
int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    TaskVariantRegistrar registrar(COMPUTE_BLOCK_TASK_ID, "compute_block_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<compute_block_task>(registrar, "compute_block_task");
  }

  return Runtime::start(argc, argv);
}
