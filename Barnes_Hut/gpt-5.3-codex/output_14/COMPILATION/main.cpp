/* ============================================================================
 * Barnes-Hut main (thread-parallel)
 *
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--threads=<NUM>]
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH = 100.0;
constexpr double LENGTH = 100.0;
constexpr double HEIGHT = 100.0;

constexpr double ORIGIN_X = 0.0;
constexpr double ORIGIN_Y = 0.0;
constexpr double ORIGIN_Z = 0.0;

static bool parse_int(std::string const& s, int& value)
{
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    value = static_cast<int>(v);
    return true;
}

template <typename F>
static void parallel_for_indices(int begin, int end, std::size_t workers, F&& body)
{
    if (begin >= end) {
        return;
    }

    workers = std::max<std::size_t>(1, workers);
    std::size_t total = static_cast<std::size_t>(end - begin);
    std::size_t block = (total + workers - 1) / workers;

    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (std::size_t w = 0; w < workers; ++w) {
        int s = begin + static_cast<int>(w * block);
        if (s >= end) {
            break;
        }
        int e = std::min(end, s + static_cast<int>(block));

        threads.emplace_back([s, e, &body]() {
            for (int idx = s; idx < e; ++idx) {
                body(idx);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

static void print_usage()
{
    std::cerr << "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM> "
                 "[--threads=<NUM>] [--hpx:threads=<NUM>]\n";
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    int requested_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto parse_prefixed_value = [&](std::string const& prefix, int& out) -> bool {
            if (arg.rfind(prefix, 0) == 0) {
                return parse_int(arg.substr(prefix.size()), out);
            }
            return false;
        };

        if (arg == "--nparticles") {
            if (i + 1 >= argc || !parse_int(argv[++i], npart)) {
                print_usage();
                return 1;
            }
        } else if (parse_prefixed_value("--nparticles=", npart)) {
        } else if (arg == "--nsteps") {
            if (i + 1 >= argc || !parse_int(argv[++i], t_step)) {
                print_usage();
                return 1;
            }
        } else if (parse_prefixed_value("--nsteps=", t_step)) {
        } else if (arg == "--threads" || arg == "--hpx:threads") {
            if (i + 1 >= argc || !parse_int(argv[++i], requested_threads)) {
                print_usage();
                return 1;
            }
        } else if (parse_prefixed_value("--threads=", requested_threads)) {
        } else if (parse_prefixed_value("--hpx:threads=", requested_threads)) {
        } else if (arg.rfind("--hpx:", 0) == 0) {
            // Ignore other HPX-specific options for compatibility.
        } else {
            print_usage();
            return 1;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        print_usage();
        return 1;
    }

    std::size_t workers = static_cast<std::size_t>(
        requested_threads > 0 ? requested_threads : static_cast<int>(std::thread::hardware_concurrency()));
    if (workers == 0) {
        workers = 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);

        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &particle_array[j]);
            }
        }

        parallel_for_indices(0, npart, workers, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for_indices(0, npart, workers, [&](int idx) {
            if (in_space(space, particle_array[idx])) {
                compute_force(&particle_array[idx], octree);
            }
        });

        parallel_for_indices(0, npart, workers, [&](int idx) {
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(&particle_array[idx]);
            }
        });

        free_octree(octree);

        print_summary(
            particle_array.data(),
            npart,
            space,
            step,
            static_cast<int>(workers));
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
