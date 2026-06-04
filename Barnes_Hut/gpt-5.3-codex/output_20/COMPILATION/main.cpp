/* ============================================================================
 * Barnes-Hut main
 * ============================================================================ */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH  = 100.0;
constexpr double LENGTH = 100.0;
constexpr double HEIGHT = 100.0;

constexpr double ORIGIN_X = 0.0;
constexpr double ORIGIN_Y = 0.0;
constexpr double ORIGIN_Z = 0.0;

namespace
{
bool parse_int(std::string const& s, int& out)
{
    try {
        std::size_t pos = 0;
        long long v = std::stoll(s, &pos);
        if (pos != s.size()) return false;
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_option(int& i, int argc, char* argv[], std::string const& name, int& out)
{
    std::string arg(argv[i]);
    if (arg == name) {
        if (i + 1 >= argc) return false;
        ++i;
        return parse_int(argv[i], out);
    }

    std::string prefix = name + "=";
    if (arg.rfind(prefix, 0) == 0) {
        return parse_int(arg.substr(prefix.size()), out);
    }

    return true;
}

void print_usage()
{
    std::fprintf(stderr,
        "Usage: ./barnes-hut [--nparticles <NUM>] [--nsteps <NUM>]\n");
}
}    // namespace

template <typename F>
void run_parallel_phase(int npart, std::size_t ntasks, F const& fn)
{
    if (npart <= 0) return;

    ntasks = std::max<std::size_t>(1, std::min<std::size_t>(ntasks, static_cast<std::size_t>(npart)));
    int chunk = (npart + static_cast<int>(ntasks) - 1) / static_cast<int>(ntasks);

    std::vector<std::thread> workers;
    workers.reserve(ntasks);

    for (std::size_t t = 0; t < ntasks; ++t) {
        int begin = static_cast<int>(t) * chunk;
        int end = std::min(npart, begin + chunk);
        if (begin >= end) break;

        workers.emplace_back([begin, end, &fn]() {
            for (int idx = begin; idx < end; ++idx) {
                fn(idx);
            }
        });
    }

    for (auto& th : workers) {
        th.join();
    }
}

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg.rfind("--hpx:", 0) == 0) {
            continue;
        }

        if (arg == "--nparticles" || arg.rfind("--nparticles=", 0) == 0) {
            if (!parse_option(i, argc, argv, "--nparticles", npart)) {
                print_usage();
                return 1;
            }
            continue;
        }

        if (arg == "--nsteps" || arg.rfind("--nsteps=", 0) == 0) {
            if (!parse_option(i, argc, argv, "--nsteps", t_step)) {
                print_usage();
                return 1;
            }
            continue;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        print_usage();
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    std::srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    std::size_t ntasks = static_cast<std::size_t>(std::thread::hardware_concurrency());
    if (ntasks == 0) ntasks = 1;

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &particle_array[j]);
            }
        }

        run_parallel_phase(npart, ntasks, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        run_parallel_phase(npart, ntasks, [&](int idx) {
            if (in_space(space, particle_array[idx])) {
                compute_force(&particle_array[idx], octree);
            }
        });

        run_parallel_phase(npart, ntasks, [&](int idx) {
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(&particle_array[idx]);
            }
        });

        free_octree(octree);

        print_summary(particle_array.data(), npart, space, step, static_cast<int>(ntasks));
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
